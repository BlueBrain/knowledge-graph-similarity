import numpy as np
from tmd.Topology import vectorizations


class Vectorisation:

    PERSISTENCE_IMAGE_RESOLUTION = 100
    FLATTEN_NORMALIZE = True

    @staticmethod
    def diagram_to_persistence_points(diagram):
        lower_points = np.array([
            [s, s - t] for s, t in diagram if s >= t
        ])
        upper_points = np.array([
            [s, t - s] for s, t in diagram if s <= t
        ])
        return lower_points, upper_points

    @staticmethod
    def evaluate_composed_density(points, x, width):
        centers = points[:, 0]
        masses = points[:, 1]
        return np.array([Vectorisation.kernel_density(el, centers, masses, width) for el in x])

    @staticmethod
    def kernel_density(x, centers, masses, kernel_width):
        density = np.sum(
            masses * np.exp(- (2 * kernel_width) ** -2 * (x - centers) ** 2))
        return density

    @staticmethod
    def compute_persistence_vector(diagram, dim, max_time, kernel_width, max_height):

        if not dim % 2:
            lower_dim = upper_dim = int(dim / 2)
        else:
            lower_dim = int(dim / 2) + 1
            upper_dim = dim - lower_dim

        lower_points, upper_points = Vectorisation.diagram_to_persistence_points(diagram)

        if lower_points.shape[0] == 0:
            lower_vector = np.zeros(lower_dim)
        else:
            lower_vector = Vectorisation.evaluate_composed_density(
                lower_points, np.linspace(0, max_time, num=lower_dim), kernel_width
            )

        if upper_points.shape[0] == 0:
            upper_vector = np.zeros(upper_dim)
        else:
            upper_vector = Vectorisation.evaluate_composed_density(
                upper_points, np.linspace(0, max_time, num=upper_dim), kernel_width
            )

        return np.concatenate([lower_vector, upper_vector])

    @staticmethod
    def persistence_image_data(**kwargs):

        xlim = kwargs["xlim"]
        ylim = kwargs["ylim"]

        bw_method = None
        weights = None

        def fc(ph):
            temp = vectorizations.persistence_image_data(
                ph, xlim=xlim, ylim=ylim, bw_method=bw_method, weights=weights,
                resolution=Vectorisation.PERSISTENCE_IMAGE_RESOLUTION
            )
            if not Vectorisation.FLATTEN_NORMALIZE:
                return temp.tolist()

            normalized = temp/temp.max()  # should occur in image_diff_data
            normalized_flattened = list(normalized.flatten())
            return normalized_flattened

        return fc

    @staticmethod
    def betti_curve(**kwargs):

        bins = None
        num_bins = 500

        return lambda ph: vectorizations.betti_curve(
            ph, bins=bins, num_bins=num_bins
        )[0]

    @staticmethod
    def life_entropy_curve(**kwargs):
        bins = None
        num_bins = 500

        return lambda ph: vectorizations.life_entropy_curve(
            ph, bins=bins, num_bins=num_bins
        )[0]
