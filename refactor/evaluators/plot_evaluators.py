import matplotlib.pyplot as plt
import numpy as np

from base.base_evaluator import BaseEvaluator
from matplotlib.backends.backend_pdf import PdfPages

class GridPlottingEvaluator(BaseEvaluator):
    """
    A simple class to plot examples from the test
    dataset with predictions.
    """
    def __init__(self, model, data, config):
        super(GridPlottingEvaluator, self).__init__(model, data, config)
        
    def evaluate(self):

        if self.config.samples > len(self.data.y_test):
            self.config.samples = len(self.data.y_test)

        plots_per_page = self.config.ncols * self.config.nrows
        total_pages = self.config.samples // plots_per_page
        colors = { i:np.random.randint(0, 255, 3) for i in np.unique(self.data.y_test) }
        print(colors)
        
        with PdfPages(self.config.output_name) as pdf:
            fig, axs = plt.subplots(
                figsize=(8, 11),
                nrows=self.config.nrows,
                ncols=self.config.ncols,
                sharex=True, sharey=True,
                dpi=self.config.dpi
            )
            fig.subplots_adjust(wspace=0, hspace=0)

            indices = np.random.choice(
                np.arange(len(self.data.X_test)),
                self.config.samples,
                replace=False
            )
            preds = self.model.model.predict(self.data.X_test)
            preds = np.argmax(preds, axis=1)
            
            # We want the data to be as it should look
            # this will reload the data without applying
            # the preprocessing.
            self.data.load()
        
            plot_order = np.argsort(preds)[np.sort(indices)]
            for i, index in enumerate(plot_order):
                pad = 1 + i % plots_per_page
                new_page = (pad == 1)
                
                # It is time to print out the
                # previous page of the pdf.
                if new_page and i > 0:
                    pdf.savefig(fig)
                    plt.close()
                    print("Printing new page {}/{}".format(
                        i // plots_per_page, total_pages
                    ))
                    
                row = (pad - 1) // self.config.ncols
                col = (pad - 1) % self.config.ncols
    
                # Plot an image there and remove axis
                # ticks if they exist to unblock the
                # figures and make them nicely sit
                # next to each other.
                img = self.data.X_test[index]
                x = self.color_pad(
                    img,
                    colors[self.data.y_test[index][0]]
                )
                axs[row,col].imshow(x)
                axs[row,col].set_xticklabels([])
                axs[row,col].set_yticklabels([])
                axs[row,col].set_title(preds[index])
            
            # It is possible that the last
            # page has not been printed.
            if not new_page:
                pdf.savefig(fig)
                plt.close()


    def color_pad(self, image, color=(123,123,123), pixels=2):
        h, w, c = image.shape
        new_image = np.zeros(
            (h + 2 * pixels, w + 2 * pixels, c),
            dtype=np.uint8)
        new_image[:,:,:] = color
        new_image[pixels : h + pixels, pixels : w + pixels] = image                
        return new_image
        
