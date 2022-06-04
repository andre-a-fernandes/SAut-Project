import matplotlib.pyplot as plt

def update_3D(i, line, data):
    line.set_data(data[i, :, 2], data[i, :, 0])
    line.set_3d_properties(data[i, :, 1])
    return line, 

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    fig, ax = plt.subplots()
    fig.axes(projection='3d')
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    #ax.plot(volume[ax.index, :, 2], volume[ax.index, :, 1], volume[ax.index, :, 0], '.')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def main():
    """from skimage import io
    
    struct_arr = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")    
    multi_slice_viewer(struct_arr.T)"""

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation

    t_max = 10
    steps = 100
    t = np.linspace(0, t_max, steps)
    x = np.cos(t)
    y = np.sin(t)
    z = 0.1*t

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], lw=1)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(0,1)

    def update(i):
        line.set_data(x[0:i], y[0:i])
        line.set_3d_properties(z[0:i])
        return line, 

    ani = animation.FuncAnimation(fig, update, frames=100, interval=10, blit=True)
    ani.save('abel.gif')
    plt.show()

if __name__ == '__main__':
    main()