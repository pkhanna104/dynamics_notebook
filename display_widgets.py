
import numpy as np 
import matplotlib.pyplot as plt 
import ipywidgets as widgets
from IPython.display import display,clear_output

import lds_utils, seq_utils

class PlotData:
    def __init__(self):
        self.x0 = None
        self.x0_prop = None
        self.cbar = None 
        self.cax = []
        self.dots = []
        
    def add_x0(self, x0):
        if self.x0 is not None:
            self.x0 = np.vstack(( self.x0, x0.copy()))
        else:
            self.x0 = x0.copy()
    
    def add_A(self, A):
        self.A = A.copy()
        
    def prop_x0(self, nT):
        npts = self.x0.shape[0]
        self.x0_prop = np.zeros((npts, 2, nT))
        for n in range(npts):
            for t in range(nT):
                self.x0_prop[n, :, t] = np.squeeze(np.dot(np.linalg.matrix_power(self.A, t),
                                               self.x0[n, :][:, np.newaxis])) 
    def mod_cbar(self, cmax, fig, ax, cax):
        if self.cbar is None:
            self.cbar = fig.colorbar(cax, ax=ax)
        else:
            for c in self.cax:
                c.set_clim([0., cmax])

    def add_cax(self, cax):
        self.cax.append(cax)
    
    def add_dots(self, dots):
        self.dots.append(dots)
    
    def clear_dots(self):
        for d in self.dots:
            for di in d:
                di.remove()
        self.dots = []
        self.x0 = None
    
    def clear_cax(self):
        for c in self.cax:
            c.remove()
        self.cax = []

class interative_viewer_Amatrix_plot(object):

    def __init__(self):
        self.dt = 0.01; 

        ### Dropdown + display A matrix
        self.A_matrices = dict()
        self.A_matrices['A1'] = np.mat([[.9, 0.], [0., 0.9]])
        self.A_matrices['A2'] = np.mat([[1.1, 0.], [0., 1.1]])
        self.A_matrices['A3'] = np.mat([[.9, -.1], [.1, 0.9]])
        self.A_matrices['A4'] = np.mat([[1.05, -.2], [.2, 1.05]])
        self.A_matrices['A5'] = np.mat([[.9, 0.], [0., -1.1]])

        ### Class for holding onto data 
        self.X0_data = PlotData()

        ### Make the dropdown window ####
        self.A_dropdown = widgets.Dropdown(options=['None', 'A1', 'A2', 'A3', 'A4', 'A5'], 
                                      value='None', description='A Matrix: ', disabled=False,
                                      layout={'width': 'initial'}, style={'description_width': 'initial'})
        self.A_dropdown.observe(self.linkA)

        ### Make the A display; 
        self.A_display = widgets.HTMLMath()

        ### Get the childern: 
        self.children1 = [self.A_dropdown, self.A_display]

         ### Plot + Clear button 
        self.out1 = widgets.Output()
        with self.out1:
            fig1, axes1 = plt.subplots(figsize=(5, 5))
            self.format_ax1(axes1)
            plt.show(fig1)
        self.fig = fig1 
        self.axes = axes1

        ### Clear button; 
        self.button=widgets.Button(description='Clear')
        self.button.on_click(self.clear_ax)
        self.children2 = [self.out1, self.button]

        #### Init conditions + time slider 
        self.num_ts = widgets.interactive(self.plot_x0_prop, {'manual': True}, 
                                     num_timepoints=widgets.IntSlider(min=1, max=20, step=1),
                                     style={'description_width': 'initial'})
        self.num_x0 = widgets.interactive(self.plot_x0, {'manual': True}, 
                                     initial_x0=widgets.IntSlider(min=1, max=10, step=1),
                                    style={'description_width': 'initial'});
            
        self.children3 = [self.num_x0, self.num_ts]

    def format_ax1(self, ax):
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_xlabel('$x0_0$')
        ax.set_ylabel('$x0_1$')
        ax.set_title('')

    def assemble_box(self):
        left = widgets.VBox(self.children1)
        right = widgets.VBox(self.children2)
        right_r = widgets.VBox(self.children3)
        self.box = widgets.HBox([left, right, right_r])


    ### Fcn to link the dropdown to the displayed A matrix ###
    def linkA(self, *args):
        val = self.A_dropdown.value; 
        if val == 'None':
            pass
        else:
            
            mat = self.A_matrices[val] 
            mat_str = r"$%s = \begin{bmatrix} %.1f & %.1f & \\ %.1f & %.1f \end{bmatrix}$"%(val, mat[0, 0],                                                      mat[0, 1], mat[1, 0], mat[1, 1])
            self.A_display.value = mat_str
            self.A_rot_mat = mat.copy()

    def clear_ax(self, *args):
        with self.out1:
            clear_output(wait=True)
            self.X0_data.clear_cax()
            self.X0_data.clear_dots()
            fig1, axes1 = plt.subplots(figsize=(5, 5))
            self.format_ax1(axes1)
            plt.show(fig1)
        self.fig = fig1; 
        self.axes = axes1; 

    def plot_x0(self, initial_x0, *args):
        num_pts = int(initial_x0)
        x0 = np.random.rand(num_pts, 2)
        x0 = 2*(x0 - 0.5)
        self.X0_data.add_x0(x0)
        with self.out1:
            clear_output(wait=True)
            self.format_ax1(self.axes)
            dots = self.axes.plot(x0[:, 0], x0[:, 1], 'k.')
            self.X0_data.add_dots(dots)
            display(self.fig)

    def plot_x0_prop(self, num_timepoints, *args):
        ## get A matrix
        mat = self.A_rot_mat
        self.X0_data.add_A(mat)
        self.X0_data.prop_x0(int(num_timepoints))
        npts = self.X0_data.x0_prop.shape[0]
        with self.out1:
            clear_output(wait=True)
            self.X0_data.clear_cax()
            
            for n in range(npts):
                cax = self.axes.scatter(self.X0_data.x0_prop[n, 0, :], 
                                    self.X0_data.x0_prop[n, 1, :], 
                                    c=np.arange(int(num_timepoints)), 
                                    s=5, cmap='viridis')
                self.X0_data.add_cax(cax)
            #X0_data.mod_cbar(num_timepoints, fig1, axes1, cax)
            display(self.fig)
            
class interactive_viewer_Amat_pls_eigs(interative_viewer_Amatrix_plot):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.eig_lines = None; 
        self.eig_dots = None; 
        
        ### Adjust child 3 to include an eigenspec; 
         ### Plot + Clear button 
        self.out2 = widgets.Output()
        with self.out2:
            fig2, axes2 = plt.subplots(figsize=(5, 2.5))
            axes2.set_title('Eigenvalue Distribution')
            axes2.set_xlabel('Frequency (Hz)')
            axes2.set_ylabel('Time Decay (sec)')
            plt.show(fig2)
        
        self.fig2 = fig2
        self.axes2 = axes2
        self.children3.append(self.out2)

        self.A_dropdown.observe(self.link_to_eig)

    def link_to_eig(self, *args):
        
        cont = True; 

        ### Get A matrix;      
        if hasattr(self.A_dropdown, 'value'):
            val = self.A_dropdown.value; 
            if val == 'None':
                cont = False; 

        if cont:
            with self.out2:
                ### Clear the axis ### 
                clear_output(wait=True)
                
                ### Remove lines / dots; 
                if self.eig_lines is not None:
                    for l in self.eig_lines:
                        l.remove()
                    for d in self.eig_dots:
                        for di in d:
                            di.remove()

                ### Plot the eigenspec
                lines, dots = lds_utils.eigenspec(self.A_rot_mat, dt = self.dt, 
                    axi = self.axes2, skip_legend=False)
                self.format_eig_axes(self.axes2)

                display(self.fig2)
                
                self.eig_lines = lines; 
                self.eig_dots = dots; 

    def format_eig_axes(self, ax):
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel ('Time Decay (sec)')
        ax.hlines(0, -10, 10, 'k', linestyle='dashed')
        ax.vlines(0, -.2, .2, 'k', linestyle='dashed')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-.2, .2])

class rotator(interactive_viewer_Amat_pls_eigs):
    def __init__(self):
        super().__init__()

        ### Modify the A Matrix settings; 
        ### Make the dropdown window ####
        self.A_dropdown = widgets.interactive(self.linkA, {'manual': True}, 
                                      rot_freq = widgets.IntSlider(min=1, max=20, step=1),
                                      layout={'width': 'initial'}, style={'description_width': 'initial'})
        self.children1[0] = self.A_dropdown
        
    def linkA(self, rot_freq=1., *args):
        rlambda = 0.98
        rot_rad = rot_freq * 2*np.pi * self.dt ## cycles / sec --> rad/sec --> rad / timestep 
        mat = rlambda*np.mat([[np.cos(rot_rad), -1*np.sin(rot_rad)],[np.sin(rot_rad), np.cos(rot_rad)]])
        mat_str = r" %s Hz rotation matrix = $\begin{bmatrix} %.1f & %.1f & \\ %.1f & %.1f \end{bmatrix}$"%(rot_freq, 
            mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1])
        self.A_display.value = mat_str
        self.A_rot_mat = mat.copy()
        self.link_to_eig()
    
    def format_eig_axes(self, ax):
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel ('Time Decay (sec)')
        ax.set_xlim([-20, 20])
        ax.hlines(0, -20, 20, 'k', linestyle='dashed')
        ax.vlines(0, -0.1, 1., 'k', linestyle='dashed')

#### For making fake data ###
class fake_data_maker(object):

    def __init__(self, *args):
        
        self.dt = 0.01; 
        self.f1 = 1.;
        self.f2 = 1.; 
        self.f3 = 1.;
        self.d1 = .1; 
        self.d2 = .5; 
        self.d3 = 1.; 
        self.noise = 0.2; 
        self.eig_lines = None;
        self.eig_dots = None;
        self.ax = None

        ### Make slider for frequency content
        layout = widgets.Layout(width='300px')

        self.freqs = widgets.interactive(self.save_freq, {'manual': True}, 
                                     f1 = widgets.IntSlider(min=1, max=20, step=1, layout=layout),
                                     f2 = widgets.IntSlider(min=1, max=20, step=1, layout=layout),
                                     f3 = widgets.IntSlider(min=1, max=20, step=1, layout=layout),
                                     style={'description_width': 'initial'})
        
        # self.decays = widgets.interactive(self.save_decay, {'manual': True}, 
        #                              d1 = widgets.FloatSlider(min=0.01, max=1., step=.01, layout=layout),
        #                              d2 = widgets.FloatSlider(min=0.01, max=1., step=.01, layout=layout),
        #                              d3 = widgets.FloatSlider(min=0.01, max=1., step=.01, layout=layout),
        #                              style={'description_width': 'initial'})
        
        self.noise_wid = widgets.interactive(self.save_noise, {'manual': True},
            noise = widgets.FloatSlider(.2, min=0.1, max=1., step=.01, layout=layout),
            style = {'description_width': 'initial'})

        self.ylim_wid = widgets.interactive(self.set_ylim_eig,
            ymax = widgets.FloatSlider(1., min=0.1, max=5., step=.01, layout=layout),
            style = {'description_width': 'initial'})

        self.r2 = widgets.HTMLMath()

        ### Eigenvalue spec plot ####
        self.out1 = widgets.Output()
        with self.out1:
            fig1, axes1 = plt.subplots(figsize=(6, 4))
            #self.format_ax1(axes1)
            plt.show(fig1)
        self.fig1 = fig1 
        self.axes1 = axes1

        # ### Data ####
        self.out2 = widgets.Output()
        with self.out2:
            fig2, axes2 = plt.subplots(ncols = 3, nrows = 3, figsize=(16, 6))
            #self.format_ax2(axes2)
            plt.show(fig2)
        self.fig2 = fig2 
        self.axes2 = axes2

        ### First row; 
        self.grandchild1 = [self.noise_wid, self.ylim_wid, self.r2]
        self.grand_row = widgets.VBox(self.grandchild1)

        self.children1 = [self.freqs, self.grand_row, self.out1]

    def assemble_box(self):
        row1 = widgets.HBox(self.children1)
        #self.children2 = [row1, self.out1]

        #col1 = widgets.VBox(self.children2)
        self.box = widgets.VBox([row1, self.out2])

        self.generate_data()

    def save_freq(self, f1, f2, f3):
        self.f1, self.f2, self.f3 = (f1, f2, f3)
        self.generate_data()

    def save_decay(self, d1, d2, d3):
        self.d1, self.d2, self.d3 = (d1, d2, d3)
        self.generate_data()
        
    def save_noise(self, noise):
        self.noise = noise; 
        self.generate_data()
        
    def generate_data(self):

        ## Generate fake data; 
        nT = 1000; ## Number of data points 

        ### Time axis: 
        self.T = np.arange(0., self.dt*nT, self.dt)

        ## Storage for components used to create data: 
        X = []; 

        ## Generate a few sine waves:
        for f in[self.f1, self.f2, self.f3]:
            s1 = np.sin(2*np.pi*f*self.T)
            c1 = np.cos(2*np.pi*f*self.T)

            ## Add them to X
            X.append(s1)
            X.append(c1)

        ## Generate a few exponential decays: 
        for r in [self.d1, self.d2, self.d3]:
            exp_dec = r**self.T
            X.append(exp_dec)

        ### Now X is full of oscillating and decaying waveforms ###
        X = np.vstack((X)).T
        assert(X.shape[0] == nT)

        _, nD = X.shape

        ### Let's add a bit of noise
        X += self.noise*np.random.randn(nT, nD) 

        ### Now lets randomly mix these waveforms together ##
        Mixing = np.random.randn(nD, nD)
        self.Data_mixed = np.dot(Mixing, X.T).T

        ### Make eigenspec; 
        self.est_A()

        ### Plot eigenspec;
        self.link_to_eig()

        ### Plot data 
        self.plot_data()

    def est_A(self):
        ## Setup X_t, X_{t-1}; 
        Xt = self.Data_mixed[1:, :] ## All data points except first
        Xtm1 = self.Data_mixed[:-1, :] ## All data points except last 

        ## Fit an A matrix using least squares linear regressions: 
        Aest = np.linalg.lstsq(Xtm1, Xt, rcond=None)[0] 

        ### This function solves for A: x_t = x_t-1 A, so we take the transpose
        Aest = Aest.T
        self.Aest = Aest; 
        self.Xtm1 = Xtm1; 

        ### Update the R2 value; 
        r2 = lds_utils.get_population_R2(Xt, np.dot(Aest, Xtm1.T).T)
        self.r2.value = r"$ R^2 = %.2f$"%(r2)      

    def plot_data(self):
        if self.ax is None:
            pass
        else:
            for a in self.ax:
                for ai in a: 
                    ai.remove()
        self.ax = []

        with self.out2:
            clear_output(wait=True)
            nT, nD = self.Data_mixed.shape

            for n in range(nD):
                ax = self.axes2[int(n/3), n%3].plot(self.T, self.Data_mixed[:, n], 'b-',
                    linewidth = .5)
                self.ax.append(ax)

                self.axes2[int(n/3), n%3].set_title('Dim %d' %n)
            self.fig2.tight_layout()
            display(self.fig2)

    
    def link_to_eig(self, *args):
        ymax = 0. 

        with self.out1:
            ### Clear the axis ### 
            clear_output(wait=True)
            
            ### Remove lines / dots; 
            if self.eig_lines is not None:
                for l in self.eig_lines:
                    l.remove()
                for d in self.eig_dots:
                    for di in d:
                        di.remove()

            ### Plot the eigenspec
            lines, dots = lds_utils.eigenspec(self.Aest, dt = self.dt, 
                axi = self.axes1, skip_legend=False)
            self.format_eig_axes(self.axes1)

            display(self.fig1)
            
            self.eig_lines = lines; 
            self.eig_dots = dots; 

    def format_eig_axes(self, *args):
        self.axes1.set_xlim([-21, 21])
        self.axes1.hlines(0, -21, 21, 'k', linestyle='dashed')
        self.axes1.vlines(0, -21, 21, 'k', linestyle='dashed')
        self.axes1.set_xlabel('Frequency (Hz)')
        self.axes1.set_ylabel('Time Decay (sec)')

    def set_ylim_eig(self, ymax):
        with self.out1:
            clear_output(wait=True)
            self.axes1.set_ylim([-.1, ymax])
            display(self.fig1)
        
class fake_data_flows(fake_data_maker):
    
    def __init__(self, *args):
        super().__init__()
        
        layout = widgets.Layout(width='350px')
        self.flow_arrows = None
        self.flow_data = None
        self.cmax0 = 1.
        self.cmax1 = 1.
        self.cmax2 = 1.


        ### Add some slider for dimension: 
        self.dims_plt0 = widgets.interactive(self.get_dims0,
                             dim0 = widgets.IntSlider(0, min=0, max=8, step=1, layout=layout),
                             dim1 = widgets.IntSlider(1, min=0, max=8, step=1, layout=layout),
                             cmax0 = widgets.FloatSlider(1., min=0.1, max=2.0, step=.01, layout=layout),
                             style={'description_width': 'initial'})
        
        self.dims_plt1 = widgets.interactive(self.get_dims1, 
                             dim2 = widgets.IntSlider(2, min=0, max=8, step=1, layout=layout),
                             dim3 = widgets.IntSlider(3, min=0, max=8, step=1, layout=layout),
                             cmax1 = widgets.FloatSlider(1., min=0.1, max=2.0, step=.01, layout=layout),
                             style={'description_width': 'initial'})
        
        self.dims_plt2 = widgets.interactive(self.get_dims2, 
                             dim4 = widgets.IntSlider(4, min=0, max=8, step=1, layout=layout),
                             dim5 = widgets.IntSlider(5, min=0, max=8, step=1, layout=layout),
                             cmax2 = widgets.FloatSlider(1., min=0.1, max=2.0, step=.01, layout=layout),
                             style={'description_width': 'initial'})

        with self.out2:
            # Clear figs #
            clear_output(wait=True)
            fig2, axes2 = plt.subplots(ncols = 3, figsize=(15, 5))
            plt.show(fig2)
        self.fig2 = fig2 
        self.axes2 = axes2

    def assemble_box(self):
        row1 = widgets.HBox(self.children1)
        
        self.children2 = [self.dims_plt0, self.dims_plt1, self.dims_plt2]
        row2 = widgets.HBox(self.children2)

        #col1 = widgets.VBox(self.children2)
        self.box = widgets.VBox([row1, row2, self.out2])
        self.generate_data()

    def get_dims0(self, dim0, dim1, cmax0):
        self.dim0 = dim0; 
        self.dim1 = dim1; 
        self.cmax0 = cmax0; 
        self.plot_data()

    def get_dims1(self, dim2, dim3, cmax1):
        self.dim2 = dim2
        self.dim3 = dim3
        self.cmax1 = cmax1; 
        self.plot_data()

    def get_dims2(self, dim4, dim5, cmax2):
        self.dim4 = dim4; 
        self.dim5 = dim5; 
        self.cmax2 = cmax2; 
        self.plot_data()

    def plot_data(self):
        ### Instead of eigspec, plot the flow fields; 
        with self.out2:

            ### Clear the axis ### 
            clear_output(wait=True)
            
            ### Remove lines / dots; 
            if self.flow_arrows is not None:
                for l in self.flow_arrows:
                    if type(l) is list:
                        for li in l:
                            li.remove()
                    else:
                        l.remove()

            if self.flow_data is not None:
                for d in self.flow_data:
                    if type(d) is list:
                        for di in d:   
                            if type(di) is list:
                                for dii in di:
                                    dii.remove()
                            else:
                                di.remove()
                    else:
                        d.remove()

            self.flow_arrows = []
            self.flow_data = []

            D = [[self.dim0, self.dim1, self.cmax0], 
                 [self.dim2, self.dim3, self.cmax1], 
                 [self.dim4, self.dim5, self.cmax2]]

            ### Plot 3 subplots ### 
            for i_f, Ds in enumerate(D):

                ### Plot Quiver ####
                Q, D = lds_utils.flow_field_plot_top_dim(self.Aest, self.Xtm1, self.dt, 
                    dim0=Ds[0], dim1=Ds[1], ax = self.axes2[i_f], cmax = Ds[2])
                self.flow_arrows.append(Q)
                self.flow_data.append(D)

            ### Fig 2 ##
            display(self.fig2)


#### For sequence data ####
class sinuoids(object):
    def __init__(self, defaults = None, fig1 = None, axes1 = None, skip_sine_params = False):
        self.xlims = [-20, 20]
        self.dt =0.01; 
        self.eig_lines = None; 
        self.eig_dots = None;
        self.cax = None
        
        if defaults is None:
            self.freq = 1; 
            self.t_offset = .1; 
            self.N_neurons = 2
            n_max = 20
            n_min = 2
            self.noise = .2
            self.ylims = 1.
            self.eig_plot = 1

        else:
            self.freq = defaults['freq']
            self.t_offset = defaults['t_offset']
            self.N_neurons = defaults['N_neurons']
            n_max = defaults['N_neurons', 'max']
            n_min = defaults['N_neurons', 'min']
            self.noise = defaults['noise']
            self.ylims = defaults['ylims']
            self.eig_plot = defaults['eig_plot']

        layout = widgets.Layout(width='400px')
        if skip_sine_params:
            pass
        else:
            ### Add some slider for dimension:
            self.sine_params = widgets.interactive(self.get_sine_params, {'manual': True},
                             freq = widgets.IntSlider(self.freq, min=1, max=10, step=1, layout=layout),
                             t_offset = widgets.FloatSlider(self.t_offset, min=0., max=.3, step=.005, layout=layout),
                             N_neurons = widgets.IntSlider(self.N_neurons, min=n_min, max=n_max, layout=layout),
                             noise = widgets.FloatSlider(self.noise, min=0.0, max=.5, step=.05, layout=layout),
                             style={'description_width': 'initial'})

        ### Get plots; ###
        self.out1 = widgets.Output()
        with self.out1:
            if fig1 is None and axes1 is None:
                fig1, axes1 = plt.subplots(ncols = 2, figsize=(8, 4))
            #self.format_ax1(axes1)
            plt.show(fig1)
        self.fig1 = fig1 
        self.axes1 = axes1

        self.ylims_eig = widgets.interactive(self.set_ylim_eigs,
                     ylim = widgets.FloatSlider(1., min=0.25, max=10, step=.25, layout=layout),
                     xmin = widgets.FloatSlider(-21., min=-21, max=21, step=.25, layout=layout),
                     xmax = widgets.FloatSlider(21.,  min=-21, max=21, step=.25, layout=layout),
                     style={'description_width': 'initial'})

    def get_sine_params(self, freq, t_offset, N_neurons, noise):
        self.freq = freq; 
        self.t_off_max = 0.5*(1/self.freq)

        tmp = self.sine_params.kwargs_widgets
        tmp[1].max = self.t_off_max

        self.t_offset = np.min([self.t_off_max, t_offset]); 
        self.N_neurons = N_neurons
        self.noise = noise
        self.gen_sinusoid()

    def assemble_box(self):
        self.children0 = widgets.HBox([self.sine_params, self.ylims_eig])
        self.children1 = [self.children0, self.out1]
        self.box = widgets.VBox(self.children1)
        self.gen_sinusoid()

    def plot_data(self):
        with self.out1:
            clear_output(wait=True)
            nT, nD = self.data.shape
            if self.cax is not None:
                self.cax.remove()

            self.cax = self.axes1[0].pcolormesh(np.arange(nT+1)*self.dt, np.arange(nD+1), 
                self.data.T, cmap='binary', vmin=-1., vmax=1.)
            self.axes1[0].set_xlabel('Time (sec)')
            self.axes1[0].set_ylabel('Neurons')
            self.axes1[0].set_xlim([0., nT*self.dt])
            self.axes1[0].set_ylim([0., nD])

            ### Function for future plotting to take hold: 
            self.more_plotting()

            ### Eig plot ##
            ### Remove lines / dots; 
            if self.eig_lines is not None:
                for l in self.eig_lines:
                    l.remove()
                for d in self.eig_dots:
                    for di in d:
                        di.remove()

            ### Plot the eigenspec
            lines, dots = lds_utils.eigenspec(self.Aest, dt = self.dt, 
                axi = self.axes1[self.eig_plot], skip_legend=False)
            self.format_eig_axes(self.axes1)
            
            self.eig_lines = lines; 
            self.eig_dots = dots; 

            self.fig1.tight_layout()
            display(self.fig1)
    
    def more_plotting(self):
        pass

    def gen_sinusoid(self):
        ### T x N ###
        self.data = seq_utils.generate_seq(self.freq, -1, self.t_offset, total_length=None, 
            N_neurons = self.N_neurons, dt=0.01, pad_zeros = 0.0, end_pad = 0., beg_pad = 0.)
        if self.noise > 0.:
            self.data += self.noise*np.random.randn(*self.data.shape)

        self.est_A()
        self.plot_data()

    def est_A(self):
        ## Setup X_t, X_{t-1}; 
        Xt = self.data[1:, :] ## All data points except first
        Xtm1 = self.data[:-1, :] ## All data points except last 

        ## Fit an A matrix using least squares linear regressions: 
        Aest = np.linalg.lstsq(Xtm1, Xt, rcond=None)[0] 

        ### This function solves for A: x_t = x_t-1 A, so we take the transpose
        Aest = Aest.T
        self.Aest = Aest; 
        self.Xtm1 = Xtm1; 
        self.r2 = lds_utils.get_population_R2(Xt, np.dot(Aest, Xtm1.T).T)
    
    def set_ylim_eigs(self, ylim, xmin, xmax):
        self.ylims = ylim; 
        self.xlims = [xmin, xmax]
        self.plot_data(); 

    def format_eig_axes(self, *args):
        self.axes1[self.eig_plot].hlines(0, -100, 100, 'k', linestyle='dashed')
        self.axes1[self.eig_plot].vlines(0, -100, 100, 'k', linestyle='dashed')
        self.axes1[self.eig_plot].set_xlim(self.xlims)
        self.axes1[self.eig_plot].set_ylim([-0.1, self.ylims])
        self.axes1[self.eig_plot].set_xlabel('Frequency (Hz)')
        self.axes1[self.eig_plot].set_ylabel('Time Decay (sec)')


class mono_sequ2(sinuoids):
    def __init__(self, *args, **kwargs):
        self.scatter = None
        self.vlines = None
        self.frac_sine_cycle = 0.5
        
        defaults = dict()
        defaults['freq'] = 2
        defaults['t_offset'] = 0.05; 
        defaults['N_neurons'] = 2; 
        defaults['N_neurons', 'max'] = 2; 
        defaults['N_neurons', 'min'] = 2; 
        defaults['noise'] = 0.
        defaults['ylims'] = 0.5
        defaults['eig_plot'] = 2
        fig1, axes1 = plt.subplots(ncols = 3, figsize=(12, 4))

        super().__init__(defaults, fig1, axes1, *args, **kwargs)
        

    def gen_sinusoid(self):
        ### T x N ###
        cycle75perc = self.frac_sine_cycle*(1/self.freq)
        self.data = seq_utils.generate_seq(self.freq, cycle75perc, self.t_offset, total_length=None, 
            N_neurons = self.N_neurons, dt=0.01, pad_zeros = 0.0, end_pad = 0., beg_pad = 0.)
        if self.noise > 0.:
            self.data += self.noise*np.random.randn(*self.data.shape)
        self.est_A()
        self.plot_data()

    def more_plotting(self):
        ax = self.axes1[1]
        
        ### Color is time steps
        if self.scatter is not None:
            for c in self.scatter:
                c.remove()
                
        cax = ax.scatter(self.data[:, 0], self.data[:, 1], s=None, 
            c=np.arange(self.data.shape[0]))
        cbar = self.fig1.colorbar(cax, ax = ax)
        cbar.set_label('Time Steps', rotation=270)
        ax.set_xlabel('Neuron 0')
        ax.set_ylabel('Neuron 1')
        self.scatter = [cbar, cax]; 

        ev, _ = np.linalg.eig(self.Aest)
        ang = np.angle(ev)/(2*np.pi*self.dt)

        self.axes1[self.eig_plot].set_title('Freq = %.2f Hz \n $R^2$ of $A_{est} = $ %.2f'%(ang[0], self.r2))

class ploy_sequ2(mono_sequ2):
    def __init__(self):
        super().__init__()
        self.frac_sine_cycle = 2.0

class flex_sequ(mono_sequ2):
    def __init__(self):
        super().__init__(skip_sine_params = True)
        self.frac_sine_cycle = 0.5

        layout = widgets.Layout(width='400px')
        ### Add some slider for dimension:
        n_min = 2; n_max = 20; 
        self.sine_params = widgets.interactive(self.get_sine_params, {'manual': True},
                             freq = widgets.IntSlider(self.freq, min=1, max=20, step=1, layout=layout),
                             t_offset = widgets.FloatSlider(self.t_offset, min=0., max=.3, step=.005, layout=layout),
                             N_neurons = widgets.IntSlider(self.N_neurons, min=n_min, max=n_max, layout=layout),
                             noise = widgets.FloatSlider(self.noise, min=0.0, max=.5, step=.05, layout=layout),
                             frac_sine_cycle = widgets.FloatSlider(self.frac_sine_cycle, min=0.1, max=3, step=.05, layout=layout),
                             style={'description_width': 'initial'})

    def get_sine_params(self, freq, t_offset, N_neurons, noise, frac_sine_cycle):
        self.freq = freq; 
        self.t_off_max = 0.5*(1/self.freq)

        tmp = self.sine_params.kwargs_widgets
        tmp[1].max = self.t_off_max

        self.frac_sine_cycle = frac_sine_cycle
        self.t_offset = np.min([self.t_off_max, t_offset]); 
        self.N_neurons = N_neurons
        self.noise = noise
        self.gen_sinusoid()

class flex_sequ_pls_flow(flex_sequ):
    def __init__(self):
        super().__init__()

        self.out2 = widgets.Output()
        with self.out2:
            fig2, axes2 = plt.subplots(ncols = 3, figsize=(15, 5))
            #self.format_ax1(axes1)
            plt.show(fig2)
        self.fig2 = fig2 
        self.axes2 = axes2

        layout = widgets.Layout(width='350px')
        self.flow_arrows = None
        self.flow_data = None
        self.cmax0 = 1.
        self.cmax1 = 1.
        self.cmax2 = 1.

        ### Add some slider for dimension: 
        self.dims_plt0 = widgets.interactive(self.get_dims0,
                             dim0 = widgets.IntSlider(0, min=0, max=8, step=1, layout=layout),
                             dim1 = widgets.IntSlider(1, min=1, max=8, step=1, layout=layout),
                             cmax0 = widgets.FloatSlider(1., min=0.1, max=2.0, step=.01, layout=layout),
                             style={'description_width': 'initial'})
        
        self.dims_plt1 = widgets.interactive(self.get_dims1, 
                             dim2 = widgets.IntSlider(2, min=0, max=8, step=1, layout=layout),
                             dim3 = widgets.IntSlider(3, min=1, max=8, step=1, layout=layout),
                             cmax1 = widgets.FloatSlider(1., min=0.1, max=2.0, step=.01, layout=layout),
                             style={'description_width': 'initial'})
        
        self.dims_plt2 = widgets.interactive(self.get_dims2, 
                             dim4 = widgets.IntSlider(4, min=0, max=8, step=1, layout=layout),
                             dim5 = widgets.IntSlider(5, min=1, max=8, step=1, layout=layout),
                             cmax2 = widgets.FloatSlider(1., min=0.1, max=2.0, step=.01, layout=layout),
                             style={'description_width': 'initial'})
    def assemble_box(self):
        self.children0 = widgets.HBox([self.sine_params, self.ylims_eig])
        dims = widgets.HBox([self.dims_plt0, self.dims_plt1, self.dims_plt2])
        self.children1 = [self.children0, self.out1, dims, self.out2]
        self.box = widgets.VBox(self.children1)
        self.gen_sinusoid()

    def get_dims0(self, dim0, dim1, cmax0):
        self.dim0 = dim0; 
        self.dim1 = dim1; 
        self.cmax0 = cmax0; 
        self.plot_data()

    def get_dims1(self, dim2, dim3, cmax1):
        self.dim2 = dim2
        self.dim3 = dim3
        self.cmax1 = cmax1; 
        self.plot_data()

    def get_dims2(self, dim4, dim5, cmax2):
        self.dim4 = dim4; 
        self.dim5 = dim5; 
        self.cmax2 = cmax2; 
        self.plot_data()

    def more_plotting(self):
        super().more_plotting()

        ### Plot the flow ###
        ### Instead of eigspec, plot the flow fields; 
        with self.out2:

            ### Clear the axis ### 
            clear_output(wait=True)
            
            ### Remove lines / dots; 
            if self.flow_arrows is not None:
                for l in self.flow_arrows:
                    l.remove()
            if self.flow_data is not None:
                for d in self.flow_data:
                    for di in d:   
                        if type(di) is list:
                            for dii in di:
                                dii.remove()
                        else:   
                            di.remove()

            self.flow_arrows = []
            self.flow_data = []

            D = [[self.dim0, self.dim1, self.cmax0], 
                 [self.dim2, self.dim3, self.cmax1], 
                 [self.dim4, self.dim5, self.cmax2]]

            ### Plot 3 subplots ### 
            for i_f, Ds in enumerate(D):

                ### Plot Quiver ####
                _, nD = self.Xtm1.shape

                if Ds[0] >= nD or Ds[1] >= nD: 
                    pass
                else:
                    Q, D = lds_utils.flow_field_plot_top_dim(self.Aest, self.Xtm1, self.dt, 
                        dim0=Ds[0], dim1=Ds[1], ax = self.axes2[i_f], cmax = Ds[2])
                    self.flow_arrows.append(Q)
                    self.flow_data.append(D)

            ### Fig 2 ##
            display(self.fig2)

