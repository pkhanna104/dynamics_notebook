
import numpy as np 
import matplotlib.pyplot as plt 
import ipywidgets as widgets
from IPython.display import display,clear_output

import lds_utils

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
