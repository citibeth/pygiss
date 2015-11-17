from matplotlib.backends.backend_gtk3 import *

class NavigationToolbar2GTK3xx(NavigationToolbar2GTK3):
    def _init_toolbar(self):
        self.set_style(Gtk.ToolbarStyle.ICONS)
        basedir = os.path.join(rcParams['datapath'],'images')

        toolitem = Gtk.ToolItem()
        self.insert(toolitem, -1)
        self.message = Gtk.Label()
        toolitem.add(self.message)

        toolitem = Gtk.SeparatorToolItem()
        self.insert(toolitem, -1)
        toolitem.set_draw(False)
        toolitem.set_expand(True)

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.insert( Gtk.SeparatorToolItem(), -1 )
                continue
            fname = os.path.join(basedir, image_file + '.png')
            image = Gtk.Image()
            image.set_from_file(fname)
            tbutton = Gtk.ToolButton()
            tbutton.set_label(text)
            tbutton.set_icon_widget(image)
            self.insert(tbutton, -1)
            tbutton.connect('clicked', getattr(self, callback))
            tbutton.set_tooltip_text(tooltip_text)

        self.show_all()


class FigureManagerGTK3xx(FigureManagerGTK3):

    def _get_toolbar(self, canvas):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if rcParams['toolbar'] == 'toolbar2':
            toolbar = NavigationToolbar2GTK3xx (canvas, self.window)
        else:
            toolbar = None
        return toolbar



FigureManager = FigureManagerGTK3xx
