from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import warnings;
with warnings.catch_warnings():
    warnings.simplefilter("ignore"); 
    import matplotlib.pyplot as plt
from skimage.transform import rescale




class Scribbler:
    
    def __init__(self, im, plot_cells1, plot_cells2, segment_names=None, title=None):
        
        self.im = im
        
                
        # f,ax = plt.subplots(1,1)
        # ax = plt.imshow(im, interpolation='none')
        # plt.axis('off')
        # plt.title(title)
        # plt.show(block=False)
        
        # create initial plot
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        plt.imshow(im, interpolation='none')
        plt.axis([0, im.shape[1], im.shape[0], 0])
        plt.scatter(plot_cells1[:,1],plot_cells1[:,0],facecolor='red',s=1)
        plt.scatter(plot_cells2[:,1],plot_cells2[:,0],facecolor='orange',s=1)
        plt.axis('off')
        
        
        if title:
            self.figure.canvas.manager.set_window_title(title)
    
        # disable default keybindings
        manager, canvas = self.figure.canvas.manager, self.figure.canvas  
        canvas.mpl_disconnect(manager.key_press_handler_id)
        
        # callbacks
        self.figure.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.on_mouse_down_id = self.figure.canvas.mpl_connect('button_press_event', self.on_mouse_down)
        self.figure.canvas.mpl_connect('button_release_event', self.on_mouse_up)
        
        self.is_connected = True
        
        # brush
        self.brush_radius = im.shape[0]/40  # TODO: these should be fractions of image size
        self.min_radius = 5
        self.radius_increment = 10
        self.brush_color = 'r'
        self.brush = None

        # scribbles
        if not segment_names:
            segment_names = [str(num+1) for num in range(7)]
        self.scribble_colors = 'bgcmykw'  # FIXME: could run out of colors
        self.scribbles = OrderedDict()
        for name in segment_names:
            self.scribbles[name] = []
        self.active_scribble = self.scribbles[segment_names[0]]
        self.active_scribble_color = self.scribble_colors[0]
        self.mouse_is_down = False                                          
        self.num_patches = 0
        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.ion()
        plt.show()
        self.figure.canvas.start_event_loop(timeout=-1)                                                                   
    
        
    def on_mouse_up(self, event):
        self.mouse_is_down = False
        
        
    def on_mouse_down(self, event):
        self.mouse_is_down = True
        if event.inaxes!=self.ax: return
        
        
        center = event.xdata, event.ydata
        self.add_circle_to_scribble(center)
        self.redraw()
                                                   
    
    def on_mouse_motion(self, event):
        center = (event.xdata, event.ydata)
        
        # draw the bush circle
        if self.brush:
            self.brush.center = center
        else:
            self.brush = matplotlib.patches.Circle(center, 
                                                    radius=self.brush_radius,
                                                    edgecolor=self.brush_color,
                                                    facecolor='none',
                                                    zorder=1e6)  # always on top
            #self.ax.add_patch(self.brush)
        self.redraw()
        
        # add to the scribble
        if self.mouse_is_down:
            self.add_circle_to_scribble(center)
        
        
    def on_keypress(self, event):
        if event.key in ['q', 'Q', 'escape']:
            self.quit_()
        elif event.key in ['=', 'super+=']:
            self.enlarge_brush()
        elif event.key in ['-', 'super+-']:
            self.shrink_brush()
        elif event.key == 'r':
            self.remove_circle_from_scribble()
        elif event.key == 'z':
            if self.is_connected:
                self.figure.canvas.mpl_disconnect(self.on_mouse_down_id)
                self.is_connected=False
                
            elif not self.is_connected:
                self.on_mouse_down_id = self.figure.canvas.mpl_connect('button_press_event', self.on_mouse_down)
                self.is_connected=True
            

#        elif event.key == 's':
#            self.save()
        elif event.key in [str(num+1) for num in range(len(self.scribbles))]:
            num = int(event.key) - 1
            name = self.scribbles.keys()[num]
            self.active_scribble = self.scribbles[name]
            self.active_scribble_color = self.scribble_colors[num]
                    
    
    def add_circle_to_scribble(self, center):
        circle = matplotlib.patches.Circle(center, 
                                           radius=self.brush_radius,
                                           edgecolor='none',
                                           facecolor=self.active_scribble_color)
        self.ax.add_patch(circle)
        self.active_scribble.append(circle)
        self.num_patches += 1
        self.redraw()
        
    
    def remove_circle_from_scribble(self):
        if self.active_scribble:
            last_circle = self.active_scribble.pop()
            last_circle.remove()
            self.num_patches -= 1
            self.redraw()
            
            
    def enlarge_brush(self):
        self.brush_radius += self.radius_increment
        if self.brush:
            self.brush.radius = self.brush_radius    
            self.redraw()
        
        
    def shrink_brush(self):
        self.brush_radius -= self.radius_increment
        self.brush_radius = max([self.brush_radius, self.min_radius])
        if self.brush:
            self.brush.radius = self.brush_radius    
            self.redraw()

    
    def quit_(self):
        plt.close()
        self.figure.canvas.stop_event_loop()
        
        
    def redraw(self):
        self.figure.canvas.draw()
    

    def add_circle_to_mask(self, center, radius, mask):
        if not center[0] or not center[1]:
            return
        xx, yy = np.mgrid[:self.im.shape[0], :self.im.shape[1]]
        circle_mask = (xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= radius**2
        mask[circle_mask] = 1
        
    
    def circles_to_mask(self, centers, radii):
        mask = np.zeros(self.im.shape[:2], dtype=bool)
        for center, radius in zip(centers, radii):
            self.add_circle_to_mask(center, radius, mask)
        return mask
        
    
    def get_scribble_masks(self):
        masks = {}
        for name, scribble in self.scribbles.items():
            centers = [circle.center for circle in scribble]
            radii = [circle.radius for circle in scribble]
            if centers:
                masks[name] = self.circles_to_mask(centers, radii)
        return masks
        
    
    def get_scribble_pixels(self):
        scribble_masks = self.get_scribble_masks()
        return {name: self.im[mask] for name, mask in scribble_masks.items()}

            

def color_hist(im, n_bins=64, colors='rgb'):
    range_ = (im.min(), im.max())
    r,g,b = map(np.squeeze, np.split(im[:-1],3,-1))
    plt.plot(np.histogram(r, n_bins, range_)[0], c=colors[0], linewidth=3)
    plt.plot(np.histogram(g, n_bins, range_)[0], c=colors[1], linewidth=3)
    plt.plot(np.histogram(b, n_bins, range_)[0], c=colors[2], linewidth=3)
    
    
    

    
# fn = r"C:\Users\akerm\OneDrive - Nexus365\Postdoc\Tracing_experiments\analysis_test\processed\images\CBLK1234_1C_left\CBLK1234_1C_left_1_.tif"
# im = plt.imread(fn)
# #im = rescale(im, 0.25)


# from image_preprocessing import find_cells

# cells = find_cells(im)

# scribbler = Scribbler(im,cells)
# scribble_masks = scribbler.get_scribble_masks()


# cell_mask = np.zeros(im.shape)
# cell_mask[cells[:,0].astype(int), cells[:,1].astype(int)] = 1
# #plt.scatter(cell_mask[:,1],cell_mask[:,0],facecolor='red',s=4)

# # these are the cells to keep
# scribble_cell_mask = np.logical_not(scribble_masks['1']) * cell_mask

# cells_fr = np.argwhere(scribble_cell_mask)

# %matplotlib qt

# plt.imshow(im)

# plt.scatter(cells[:,1],cells[:,0],facecolor='red',s=4)
# plt.scatter(cells_fr[:,1],cells_fr[:,0],facecolor='green',s=4)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    