import numpy as np
from tmd.Topology import vectorizations


class Vectorisation:

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
    def build_vectors_from_tmd_implementations(ph1):

            xlim = None
            ylim = None
            bw_method = None
            weights = None
            resolution = 100
            bins = None
            num_bins = 1000

            a = vectorizations.persistence_image_data(
                ph1, xlim=xlim, ylim=ylim, bw_method=bw_method, weights=weights,
                resolution=resolution
            )
            b = vectorizations.betti_curve(
                ph1, bins=bins, num_bins=num_bins
            )[0]
            c = vectorizations.life_entropy_curve(
                ph1, bins=bins, num_bins=num_bins
            )[0]

            return a, b, c