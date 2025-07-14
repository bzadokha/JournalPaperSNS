import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import List
import sns_toolbox
import sns_toolbox.networks
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.neurons import NonSpikingNeuron


def trajectoryInput(t, T, theta_list, func_type, to_plot=False):
    if func_type == 'sinusoid':
        '''
        in this case, thetamin = theta_list[0] and thetamax = theta_list[1]
        '''
        thetamin = theta_list[0]
        thetamax = theta_list[1]
        inputT = (thetamin + thetamax) / 2 + (thetamax - thetamin) / 2 * np.sin(2 * np.pi * t / T)
    elif func_type == 'steps':
        '''
        replace thetamin and thetamax with vectors
        make the signal the 0th value for T, then the 1st value for T after that, etc.
        '''
        tmax = np.max(t)
        inputT = np.zeros_like(t)
        num_steps = int(np.ceil(tmax/T))
        num_vals = np.size(theta_list)
        for i in np.arange(num_steps):
            bin_mask = np.floor((t / T)) == i
            inputT[bin_mask] = theta_list[np.mod(i, num_vals)]

    else:
        inputT = None

    if to_plot:
        qw = 1
        # plt.figure()
        # plt.plot(t, inputT)
        # plt.draw()
        # plt.show()
    return inputT

def bell_curve(magnitude, width, theta, shift):
    return magnitude * np.exp(-width * pow((theta - shift), 2))

class NonSpikingLayer(nn.Module):
    def __init__(self,
                 num_neurons: int,
                 num_inputs: int = 1,  # Used for Output layer k_syn shape
                 parameters: dict = None,
                 neuron_type: str = 'Input',
                 connection_type: str = 'row_col',
                 device: torch.device = torch.device('mps'),
                 name: str = 'Layer'):
        super().__init__()
        self.name = name
        self.neuron_type = neuron_type
        self.device = device
        self.connection_type = connection_type

        # For NSI layers, num_neurons is the size of one input dimension (N)
        if self.neuron_type == "NSI":
            self.N = num_neurons
            self.num_neurons = self.N * self.N # Total neurons in the N x N grid
            self.num_inputs = num_inputs
        else:
            self.N = num_neurons
            self.num_neurons = num_neurons
            self.num_inputs = num_inputs

        # Default parameters
        if parameters is None: parameters = {}
        self.dt = parameters.get('dt', 0.1)
        c_m = parameters.get('c_m', 1.0)
        e_rest = parameters.get('e_rest', 0.0)
        g_m_leak = parameters.get('g_m_leak', 5.0)
        e_syn = parameters.get('e_syn', 20.0)
        R_op_range = parameters.get('R_op_range', 1.0)
        i_bias = parameters.get('i_bias', 0.0)

        # Neuron parameters
        self.c_m = torch.full((self.num_neurons, 1), c_m, device=self.device)
        self.e_rest = torch.full((self.num_neurons, 1), e_rest, device=self.device)
        self.g_m_leak = torch.full((self.num_neurons, 1), g_m_leak, device=self.device)
        self.e_syn_delta = torch.full((self.num_neurons, 1), e_rest + e_syn, device=self.device)
        self.R_op_range = torch.full((self.num_neurons, 1), R_op_range, device=self.device)
        self.i_bias = torch.full((self.num_neurons, 1), i_bias, device=self.device)

        self.v_m = self.e_rest.clone().squeeze(1)

        # *** Define k_syn as a learnable parameter for backprop ***
        if self.neuron_type == "NSI":
            # For NSI, k_syn has a gain for each of the N neurons from the 2 input sources.
            initial_k_syn = torch.ones(self.num_inputs, self.N, device=self.device)
            # self.k_syn = nn.Parameter(initial_k_syn) #do not need for connections between input and NSI
            self.k_syn = initial_k_syn
        elif self.neuron_type == "Output":
            # For Output, k_syn has a gain for each connection from num_inputs to num_neurons.
            initial_k_syn = torch.ones(self.num_inputs, self.num_neurons, device=self.device)
            self.k_syn = nn.Parameter(initial_k_syn)

    def _dv_dt(self, v_post, i_ext, i_syn_values=None):
        i_leak = (v_post - self.e_rest) * self.g_m_leak
        if i_syn_values is None:
            return (self.i_bias + i_ext - i_leak) / self.c_m
        else:
            return (self.i_bias + i_ext + i_syn_values - i_leak) / self.c_m

    def g_max(self, k_syn_gain_t: torch.Tensor, num_active_inputs: int) -> torch.Tensor:
        g_max_factor_num = k_syn_gain_t * self.R_op_range.T
        g_max_factor_den = self.e_syn_delta.T - g_max_factor_num * num_active_inputs
        if torch.any(g_max_factor_den <= 1e-9):
            raise ValueError("Denominator in g_max is near zero.")
        return g_max_factor_num / g_max_factor_den

    def g_syn(self, u_pre: torch.Tensor, g_max_t: torch.Tensor) -> torch.Tensor:
        activation = torch.clamp(u_pre / self.R_op_range.T, min=0.0, max=1.0)
        return g_max_t * activation

    def _rk4_forward_pass(self, i_syn_total_values: torch.Tensor, i_ext_values: torch.Tensor):

        v_m_unsqueezed = self.v_m.unsqueeze(1)

        k1 = self._dv_dt(v_m_unsqueezed, i_ext_values, i_syn_total_values).squeeze(1)
        k2 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k1.unsqueeze(1), i_ext_values, i_syn_total_values).squeeze(1)
        k3 = self._dv_dt(v_m_unsqueezed + 0.5 * self.dt * k2.unsqueeze(1), i_ext_values, i_syn_total_values).squeeze(1)
        k4 = self._dv_dt(v_m_unsqueezed + self.dt * k3.unsqueeze(1), i_ext_values, i_syn_total_values).squeeze(1)

        self.v_m += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return self.v_m

    def _nsi_forward_pass(self, u_pre_vector):

        num_steps = u_pre_vector[0].shape[1]
        # Create a V_output tensor
        v_m_Post = torch.zeros(self.num_neurons, num_steps, device=self.device)

        # Unpack inputs and use the internal k_syn parameter
        # should think about how to make this part modifiable so we can have >2 inputs
        u_pre_0, u_pre_1 = u_pre_vector[0], u_pre_vector[1]
        k_syn_0, k_syn_1 = self.k_syn[0, :], self.k_syn[1, :]

        for t in range(num_steps):
            u_pre_0_t, u_pre_1_t = u_pre_0[:, t], u_pre_1[:, t]
            v_m_grid = self.v_m.view(self.N, self.N)

            if self.connection_type == 'row_col':
                u0_grid = u_pre_0_t.view(self.N, 1).expand(-1, self.N)
                u1_grid = u_pre_1_t.view(1, self.N).expand(self.N, -1)
                k0_grid = k_syn_0.view(self.N, 1).expand(-1, self.N)
                k1_grid = k_syn_1.view(1, self.N).expand(self.N, -1)
                g_max0 = self.g_max(k0_grid.flatten().unsqueeze(0), 1)
                g_syn0 = self.g_syn(u0_grid.flatten().unsqueeze(0), g_max0).view(self.N, self.N)
                g_max1 = self.g_max(k1_grid.flatten().unsqueeze(0), 1)
                g_syn1 = self.g_syn(u1_grid.flatten().unsqueeze(0), g_max1).view(self.N, self.N)
                i_syn_0 = g_syn0 * (self.e_syn_delta.view(self.N, self.N) - v_m_grid)
                i_syn_1 = g_syn1 * (self.e_syn_delta.view(self.N, self.N) - v_m_grid)
                i_syn_total = (i_syn_0 + i_syn_1).flatten().unsqueeze(1)

            elif self.connection_type == 'full':
                num_active_inputs = self.num_inputs * self.N
                g_max_0 = self.g_max(k_syn_0.unsqueeze(0), num_active_inputs)
                g_max_1 = self.g_max(k_syn_1.unsqueeze(0), num_active_inputs)
                g_syn_from_0 = self.g_syn(u_pre_0_t.unsqueeze(0), g_max_0)
                g_syn_from_1 = self.g_syn(u_pre_1_t.unsqueeze(0), g_max_1)
                total_g_syn_per_neuron = g_syn_from_0.sum() + g_syn_from_1.sum()
                i_syn_total = total_g_syn_per_neuron * (self.e_syn_delta - self.v_m.unsqueeze(1))

            self.v_m = self._rk4_forward_pass(i_syn_total_values=i_syn_total, i_ext_values=0)
            v_m_Post[:, t] = self.v_m

        return v_m_Post.view(self.N, self.N, num_steps)

    def _output_forward_pass(self, u_pre_vector) -> torch.Tensor:
        '''
        function will tune connections (k_syn values) between NSI and Output neurons
        using normalized(?) XYZ positions of the endpoint
        :param u_pre_vector:
        :param mapXYZ:
        :return:
        '''
        num_steps = u_pre_vector[0].shape[1]
        v_m_Post = torch.zeros(self.num_neurons, num_steps, device=self.device)

        for t in range(num_steps):
            i_ext_t = torch.zeros(self.num_neurons, 1, device=self.device)

            i_syn_total = torch.zeros(self.num_neurons, 1, device=self.device)

            u_pre_t = torch.stack([u[:, t] for u in u_pre_vector])
            k_syn_t = self.k_syn  # Use internal parameter
            num_active_inputs = len(u_pre_vector)
            g_max_t = self.g_max(k_syn_t, num_active_inputs)
            g_syn_t = self.g_syn(u_pre_t, g_max_t)
            i_syn_per_connection = g_syn_t * (self.e_syn_delta.T - self.v_m)
            i_syn_total = i_syn_per_connection.sum(dim=0, keepdim=True).T

            self.v_m = self._rk4_forward_pass(i_syn_total, i_ext_t)

            v_m_Post[:, t] = self.v_m
        return v_m_Post

    def _input_forward_pass(self, i_ext:torch.Tensor):

        num_steps = i_ext.shape[1]

        v_m_Post = torch.zeros(self.num_neurons, num_steps, device=self.device)

        for t in range(num_steps):
            i_ext_t = torch.zeros(self.num_neurons, 1, device=self.device)
            i_syn_total = torch.zeros(self.num_neurons, 1, device=self.device)

            i_ext_t = i_ext[:, t].unsqueeze(1)

            self.v_m = self._rk4_forward_pass(i_syn_total, i_ext_t)

            v_m_Post[:, t] = self.v_m

        return v_m_Post

    # *** MODIFIED: k_syn_vector removed from arguments ***
    def forward(self, i_ext_vector: torch.Tensor = None,
                u_pre_vector: List[torch.Tensor] = None) -> torch.Tensor:

        if self.neuron_type == "NSI":
            if u_pre_vector is None or len(u_pre_vector) != 2:
                raise ValueError("NSI layer requires a list of two presynaptic inputs.")
            return self._nsi_forward_pass(u_pre_vector)

        elif self.neuron_type == "Output":
            if u_pre_vector is None: raise ValueError("Output layer requires presynaptic inputs.")
            return self._output_forward_pass(u_pre_vector)

        elif self.neuron_type == "Input":
            if i_ext_vector is None: raise ValueError("Input layer requires i_ext.")
            return self._input_forward_pass(i_ext_vector)

        else:
            raise ValueError(f"Invalid neuron type: {self.neuron_type}")


if __name__ == '__main__':
    # --- Setup (Device, Parameters, Inputs) ---
    # (This part is the same as the previous version)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) accelerator.")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")
    dt = 1.0
    tMax = 3e3
    t = np.arange(0, tMax, dt)
    numSteps = len(t)
    numJoints = 2
    numNeurons = 5
    thetaMin, thetaMax = -1.6, 1.6
    jointTheta = np.linspace(thetaMin, thetaMax, numNeurons)
    c_m, width, mag = 1.0, 7, 1.0
    T1, T2 = 450, 600
    theta1vec = trajectoryInput(t, T1, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
    theta2vec = trajectoryInput(t, T2, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
    networkInputs = torch.Tensor(np.array([theta1vec, theta2vec]))
    neuron_parameters = {'c_m': c_m, 'e_rest': 0.0, 'g_m_leak': 1.0, 'e_syn': 20.0, 'dt': dt, 'R_op_range': mag}
    inputBellResponses = torch.zeros([numJoints, numNeurons, numSteps], device=device)
    for joint in range(numJoints):
        for neuron in range(numNeurons):
            inputBellResponses[joint, neuron, :] = torch.tensor(
                bell_curve(magnitude=mag, theta=networkInputs[joint, :], shift=jointTheta[neuron], width=width))

    print("Starting simulation...")
    start = time.time()
    input_layer_joint1 = NonSpikingLayer(num_neurons=numNeurons, neuron_type="Input", parameters=neuron_parameters,
                                         device=device)
    input_layer_joint2 = NonSpikingLayer(num_neurons=numNeurons, neuron_type="Input", parameters=neuron_parameters,
                                         device=device)

    # --- Create NSI Layer ---
    nsi_layer = NonSpikingLayer(num_neurons=numNeurons, neuron_type="NSI",
                                connection_type='row_col',
                                parameters=neuron_parameters, device=device, num_inputs=numJoints)

    # --- Run Input Layers ---
    v_m_out_joint1 = input_layer_joint1(i_ext_vector=inputBellResponses[0])
    v_m_out_joint2 = input_layer_joint2(i_ext_vector=inputBellResponses[1])

    # *** DEMO: Show that k_syn is a learnable parameter ***
    # print("\nNSI Layer learnable parameters:")
    # for name, param in nsi_layer.named_parameters():
    #     print(f"  Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
    # print("-" * 30)

    # --- Run NSI Layer ---
    v_m_to_nsi = [v_m_out_joint1, v_m_out_joint2]

    # *** MODIFIED: No longer pass k_syn_vector to forward() ***
    v_m_out_nsi = nsi_layer(u_pre_vector=v_m_to_nsi)

    print("\nSimulation finished.")
    print(f"--- {time.time() - start:.2f} seconds ---")

    # --- Plotting (same as previous version) ---
    # (Plotting code remains unchanged)
    # --- Plotting Results ---
    fig, axs = plt.subplots(3, 2, figsize=(22, 24), gridspec_kw={'height_ratios': [1, 1, 1.5]})
    fig.suptitle('Three-Layer Network Simulation', fontsize=16)

    # Plot joint angles
    for j in range(numJoints):
        axs[0, 0].plot(t, networkInputs[j, :], label=f'Input joint {j}')
    axs[0, 0].set_title('Joint Angles Inputs')
    axs[0, 0].set_ylabel('Angles (rad)')
    axs[0, 0].grid(True)
    # axs[0, 0].set_xlim([0, 2e3])
    axs[0, 0].legend()

    # Plot a few neurons from the NSI grid over time
    axs[0, 1].set_title('NSI Layer - Membrane Potentials')
    for i in range(0, numNeurons):
        for j in range(0, numNeurons):
            axs[0, 1].plot(t, v_m_out_nsi[i, j, :].cpu().detach().numpy(), label=f'V_m of NSI Neuron ({i},{j})')
    axs[0, 1].set_ylabel('Voltage (mV)')
    axs[0, 1].grid(True)

    # axs[0, 1].set_xlim([0, 2000])
    # axs[0, 1].legend()

    # Plot input layer 1 voltages
    # axs[1, 0].set_title('Input Layer 1 - Postsynaptic Potentials')
    # for neuron in range(0, numNeurons):
    #     axs[1, 0].plot(t, v_m_out_joint1[neuron, :].cpu().detach().numpy())
    # axs[1, 0].set_ylabel('Voltage (mV)')
    # # axs[1, 0].set_xlim([0, 2e3])
    # axs[1, 0].grid(True)
    #
    # # Plot input layer 2 voltages
    # axs[1, 1].set_title('Input Layer 2 - Postsynaptic Potentials')
    # for neuron in range(0, numNeurons):
    #     axs[1, 1].plot(t, v_m_out_joint2[neuron, :].cpu().detach().numpy())
    # axs[1, 1].set_ylabel('Voltage (mV)')
    # # axs[1, 1].set_xlim([0, 2e3])
    # axs[1, 1].grid(True)

    # Add a heatmap of NSI activity at a specific time
    time_point_to_plot = int(numSteps / 4)
    ax_heatmap = plt.subplot(212)  # Span the whole bottom row
    fig.add_axes(ax_heatmap)
    # ax_heatmap.set_position([0.125, 0.1, 0.775, 0.25])  # Manually position it
    im = ax_heatmap.imshow(v_m_out_nsi[:, :, time_point_to_plot].cpu().detach().numpy(),
                           cmap='viridis', aspect='auto', origin='lower',
                           extent=[0, numNeurons - 1, 0, numNeurons - 1])
    ax_heatmap.set_title(f'NSI Grid Activity Heatmap at t = {t[time_point_to_plot]} ms')
    ax_heatmap.set_xlabel('Neuron Index (from Input 2)')
    ax_heatmap.set_ylabel('Neuron Index (from Input 1)')
    fig.colorbar(im, ax=ax_heatmap, label='Voltage (mV)')

    # Remove the now-empty axes
    fig.delaxes(axs[1, 0])
    fig.delaxes(axs[1, 1])

    # Set shared x-limits for time plots
    for i in range(2):
        for j in range(2):
            if (i, j) not in [(2, 0), (2, 1)]:
                axs[i, j].set_xlim([0, tMax / 2])

    # plt.tight_layout(rect=[0, 0.3, 1, 0.96])
    plt.show()

'''
to do 
compare sns NSI outputs to this network - done
'''

'''
to do 
output part with NSI tuning
allow backprop for Kgain?

'''

'''



SNS



'''
netName = 'SNScheck'

net = sns_toolbox.networks.Network(name=netName)


def Gmax2(k, R, delE):
    maxConduct = k * R / (delE - 2 * k * R)
    return maxConduct


mag = 1
delEex = 20
delEin = -20
kIdent = 1
delay = 1
numJoints = 2
numNeurons = 5
gIdent = Gmax2(kIdent, mag, delEex)  # UPDATED: solve for g to make postsyn neuron U = 2*mag.

identitySyn = NonSpikingSynapse(max_conductance=gIdent, reversal_potential=delEex, e_lo=0, e_hi=mag)

bellNeuron = NonSpikingNeuron(membrane_capacitance=delay, membrane_conductance=1)

NSIneuron = NonSpikingNeuron(membrane_capacitance=delay, membrane_conductance=1, bias=0)

mag = 1
delEex = 20
delEin = -20
kIdent = 1

for joint in range(numJoints):
    # input neurons
    for neuron in range(numNeurons):
        name = 'Bell_' + str(joint + 1) + '_' + str(neuron)
        net.add_neuron(neuron_type=bellNeuron, name=name)
        net.add_input(name)
        net.add_output(name)
        print(name)

'''
    Build the SNS-Toolbox model. Add the NSI neurons in a grid. Follow the naming provided by the user. All the synapses
    from input neurons to the central grid have the same "identity" tuning.
    '''
for i in range(numNeurons):
    for j in range(numNeurons):
        name1 = 'Bell_' + str(1) + '_' + str(i)
        name2 = 'Bell_' + str(2) + '_' + str(j)

        nameStr = 'nsi' + '_' + str(i) + '_' + str(j)

        # from input sensory neurons to NSIs 1st layer
        net.add_neuron(neuron_type=NSIneuron, name=nameStr)

        net.add_connection(identitySyn, source=name1, destination=nameStr)
        try:
            net.add_connection(identitySyn, source=name2, destination=nameStr)
        except ValueError:
            print('caught an error')
        print(nameStr)
        net.add_output(source=nameStr)

model = net.compile(backend='numpy', dt=dt)
numOut = net.get_num_outputs()
PredictedOut = np.zeros([numSteps, numOut])

# inputs
T1 = 450
T2 = 600  # 99.8

theta1vec = trajectoryInput(t, T1, theta_list=[thetaMin, thetaMax], func_type='sinusoid')
theta2vec = trajectoryInput(t, T2, theta_list=[thetaMin, thetaMax], func_type='sinusoid')

networkInputs = np.squeeze(np.array([[theta1vec.transpose()], [theta2vec.transpose()]]))
'''
    Create sensory input matrices for n joints. This matrix stores bell responses for all joints
'''
inputs = np.zeros([numJoints, numSteps, numNeurons])
'''
    Create an input vector for a network of n joints. 
    It requires to 'squeeze' all inputs into one-dimensional vector by numSteps.
'''
inputNet = np.zeros([numSteps, numNeurons * numJoints])

'''
    For the network to perform the desired calculation, the synapses from each combo neuron to the output neuron should have
    effective gains (i.e., Vout/Vcombo) that are proportional to the value that the output neuron encodes. Here, we take all
    the "training data", that is, the actual X coordinate of the leg, normalize it to values between 0 and 1, and use those
    to set the unique properties of the synapse from each combo neuron to the output neuron.
'''

for joint in range(numJoints):
    for neuron in range(numNeurons):
        inputs[joint, :, neuron] = bell_curve(magnitude=mag, theta=networkInputs[joint, :], shift=jointTheta[neuron],
                                              width=width)

for i in range(numSteps):
    inputNet[i, :] = np.concatenate(inputs[:, i, :], axis=None)

for i in range(len(t)):
    PredictedOut[i, :] = model(inputNet[i, :])
print('Num outputs: ', str(net.get_num_outputs))

plt.figure()
for i in range(numNeurons * 2, numOut):
    plt.plot(t, PredictedOut[:, i])
    plt.xlim([0, tMax / 2])
plt.show()

'''
to do 
compare sns NSI outputs to this network - done
'''

'''
to do 
output part with NSI tuning
allow backprop for Kgain?

'''




