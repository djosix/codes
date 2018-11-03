#!/usr/bin/env python3


#=============================================================================
# GUI

def application():
    import tkinter as tk
    app = tk.Tk()
    app.title('SOM Settings')
    app.resizable(width=False, height=False)

    sticky = {'sticky': tk.E + tk.W + tk.S + tk.N}

    def var(text, var_type, default, row, pady=0):
        variable = var_type()
        variable.set(default)
        label = tk.Label(app)
        label['text'] = text
        label.grid(row=row, column=0, padx=[10, 0], pady=[5 + pady, 0], **sticky)
        entry = tk.Entry(app)
        entry['textvariable'] = variable
        entry.grid(row=row, column=1, padx=[5, 10], pady=[5, 0], **sticky)
        return variable

    epochs = var('Epochs', tk.IntVar, 100, 0, 5)
    batch_size = var('Batch size (0 for all)', tk.IntVar, 0, 1)
    size = var('Size', tk.StringVar, '20 20', 2)
    limit = var('Limit', tk.StringVar, '-0.2 0.2', 3)
    eta = var('Eta', tk.DoubleVar, 0.02, 4)
    tou = var('Tou', tk.StringVar, '10000 10000', 5)
    sigma = var('Sigma', tk.DoubleVar, 0.25, 6)

    from glob import glob
    file_select = tk.Listbox(app)
    file_select.insert(0, *glob('datasets/*'))
    file_select.grid(row=7, column=0, columnspan=2, padx=[10, 10], pady=[5, 0], **sticky)

    def start_animation():
        try:
            config = {
                'epochs': epochs.get(),
                'batch_size': batch_size.get(),
                'size': list(map(int, size.get().split())),
                'limit': list(map(float, limit.get().split())),
                'eta': eta.get(),
                'tou': list(map(int, tou.get().split())),
                'sigma': sigma.get(),
                'dataset': file_select.get(file_select.curselection()[0])
            }
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showinfo('Oh no!', 'Some of the settings gone wrong, e.g. dataset.\n Error: {}'.format(e))
            return
            
        try:
            ani = animation(config)
            app.after(0, ani.start)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showinfo('Oops!', 'SOM is broken?\nError: {}'.format(e))


    button = tk.Button(app)
    button['text'] = 'Start'
    button['command'] = start_animation
    button.grid(row=8, column=0, columnspan=2, padx=[10, 10], pady=[5, 10], **sticky)

    app.mainloop()


#=============================================================================
# Animation

def animation(config):
    from som import SOM
    som = SOM(config)

    from animation import App
    from pygame import Color, draw
    ani = App(name='SOM Animation', size=(800, 800), fps=None)

    def tr(p, from_range=(-1, 1), to_range=(100, 700)):
        len_from = from_range[1] - from_range[0]
        len_to = to_range[1] - to_range[0]
        ratio = len_to / len_from
        return (p - from_range[0]) * ratio + to_range[0]

    @ani.use
    def render():
        for x in som.dataset.x:
            draw.circle(ani.screen, Color('red'), tr(x).astype(int), 3)
        for i in range(som.mat.shape[0] - 1):
            for j in range(som.mat.shape[1]):
                draw.line(ani.screen, Color('black'), *map(tr, som.mat[i:i+2, j]))
        for j in range(som.mat.shape[1] - 1):
            for i in range(som.mat.shape[0]):
                draw.line(ani.screen, Color('black'), *map(tr, som.mat[i, j:j+2]))

    @ani.use
    def after_render():
        som.step()

    return ani


#=============================================================================
# Main

if __name__ == '__main__':
    application()
