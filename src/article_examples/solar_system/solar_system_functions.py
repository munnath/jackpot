import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from jackpot.utils import default_plot_styles

linestyles, colors = default_plot_styles()

class SolarModel(nn.Module):
    def __init__(
        self,
        n_planets=5,
        what_is_param=["mass", "position", "speed"],
        solar_mass_included=True,
        device=torch.device("cuda"),
        dtype=torch.float32,
        export_speed=False,
        subselect_every=1,
    ):
        super().__init__()
        # Constant values
        self.n_planets = n_planets
        self.what_is_param = what_is_param
        self.solar_mass_included = solar_mass_included
        self.device = device
        self.dtype = dtype
        self.export_speed = export_speed
        self.subselect_every = subselect_every
        self.G = 2.95912208286 / 1e4

        old_method = True

        if self.n_planets == 5 and old_method:
            self.mass = torch.tensor(
                [
                    0.000954786104043,
                    0.000285583733151,
                    0.0000437273164546,
                    0.0000517759138449,
                    1.0 / (1.3 * 10**8),
                    1.00000597682,
                ],
                device=device,
                dtype=dtype,
            )

            self.planet_names = [
                "Jupiter",
                "Saturn",
                "Uranus",
                "Neptun",
                "Pluto",
            ]

            self.X0 = np.reshape(
                np.array(
                    [
                        [-3.5023653],
                        [-3.8169847],
                        [-1.5507963],
                        [9.0755314],
                        [-3.0458353],
                        [-1.6483708],
                        [8.3101420],
                        [-16.2901086],
                        [-7.2521278],
                        [11.4707666],
                        [-25.7294829],
                        [-10.816945],
                        [-15.5387357],
                        [-25.2225594],
                        [-3.1902382],
                        [0],
                        [0],
                        [0],
                        [0.00565429],
                        [-0.00412490],
                        [-0.00190589],
                        [0.00168318],
                        [0.00483525],
                        [0.00192462],
                        [0.00354178],
                        [0.00055029],
                        [0.00055029],
                        [0.00288930],
                        [0.00039677],
                        [0.00039677],
                        [0.00276725],
                        [-0.00136504],
                        [-0.00136504],
                        [0],
                        [0],
                        [0],
                    ]
                ),
                36,
            )

            self.X0 = torch.tensor(self.X0, device=device, dtype=dtype)
        elif self.n_planets <= 10:
            mass = [
                0.16601e-6,
                2.4478383e-6,
                3.00348959632e-6,
                0.3227151e-6,
                954.79194e-6,
                285.8860e-6,
                43.66244e-6,
                51.51389e-6,
                0.007396e-6,
                0.0369430370099999e-6,
                1,
            ]
            self.mass = torch.tensor(
                mass[: self.n_planets] + [mass[-1]], device=device, dtype=dtype
            )

            self.planet_names = [
                "Mercury",
                "Venus",
                "Earth",
                "Mars",
                "Jupiter",
                "Saturn",
                "Uranus",
                "Neptun",
                "Pluto",
                "Moon",
            ]
            self.planet_names = self.planet_names[: self.n_planets]
            
            # In AU
            Pos = [
                [-0.140769373233, -0.397974073748, -0.197982763468],
                [-0.718631990376, -0.036960790815, 0.02885149967],
                [-0.168439604575, 0.888856600386, 0.385362008506],
                [1.390355565949, -0.005606444919, -0.040164617511],
                [4.003587430764, 2.733472971444, 1.074122313819],
                [6.408778864821, 6.172709081341, 2.273863973779],
                [14.430211873439, -12.507780350821, -5.682326322161],
                [16.810310184192, -22.981221680523, -9.824841028058],
                [-9.877395018927, -27.978121620359, -5.752833661466],
                [-0.170564682378, 0.887278001434, 0.38494394531],
                [0, 0, 0],
            ]
            # In AU per day !
            Speed = [
                [0.02116807009, -0.005511006181, -0.005139191809],
                [0.000511587551, -0.018508433838, -0.008359068324],
                [-0.017234198783, -0.00275811858, -0.001195752652],
                [0.000749041627, 0.013813917096, 0.006315748673],
                [-0.004563203052, 0.005884954362, 0.002633675403],
                [-0.004290991246, 0.003529657363, 0.00164241516],
                [0.002678441871, 0.002461719797, 0.001040280076],
                [0.002579401266, 0.001668261501, 0.000618745558],
                [0.003028685401, -0.001127790632, -0.001265167284],
                [-0.016910810431, -0.003181981494, -0.00138048986],
                [0, 0, 0],
            ]

            self.X0 = torch.ravel(
                torch.tensor(
                    Pos[: self.n_planets]
                    + [Pos[-1]]
                    + Speed[: self.n_planets]
                    + [Speed[-1]],
                    device=device,
                    dtype=dtype,
                )
            )

        self.init_positions = self.X0[: (self.n_planets + 1) * 3]
        self.init_speeds = self.X0[(self.n_planets + 1) * 3:]

        self.planet_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Initial position
        if self.solar_mass_included:
            self.n_masses = self.n_planets + 1
        else:
            self.n_masses = self.n_planets
        # Sun position and speed are fixed
        self.n_positions = self.n_planets * 3
        self.n_speeds = self.n_planets * 3
        self.n_params = 0
        pos_i = 0
        speed_i = 0

        if "mass" in self.what_is_param:
            self.n_params += self.n_masses

        if "position" in self.what_is_param:
            self.n_params += self.n_positions

        if "speed" in self.what_is_param:
            self.n_params += self.n_speeds

        self.param_default = torch.zeros(
            (self.n_params,), device=self.device, dtype=self.dtype
        )

        if "mass" in self.what_is_param:
            pos_i += self.n_masses
            speed_i += self.n_masses
            self.param_default[:pos_i] = self.mass[:pos_i]

        if "position" in self.what_is_param:
            speed_i += self.n_positions
            self.param_default[pos_i:speed_i] = self.init_positions[
                : self.n_positions
            ]

        if "speed" in self.what_is_param:
            self.param_default[speed_i:] = self.init_speeds[: self.n_positions]

    # Direct model
    def resolution(self, X0, m, Tinit=10000, Tfin=20000, dt=100):
        N_init = int(Tinit / dt)
        N = int(Tfin / dt)
        X = torch.zeros(
            [(self.n_planets + 1) * 3 * 2, N + 1],
            device=self.device,
            dtype=self.dtype,
        )
        X[:, 0] = X0
        for i in range(N):
            X[:, i + 1] = self.RK4(X[:, i], dt, m)
        return X[:, N_init:]

    def x_from_param(self, param):
        # torch.zeros(((self.n_planets + 1) * 6,), device=device, dtype=dtype)
        x0 = self.X0.clone()
        pos_i = 0
        speed_i = 0
        if "mass" in self.what_is_param:
            pos_i += self.n_masses
            speed_i += self.n_masses

        if "position" in self.what_is_param:
            speed_i += self.n_positions
            positions = param[pos_i:speed_i]
            x0[: (self.n_planets) * 3] = positions

        if "speed" in self.what_is_param:
            speeds = param[speed_i:]
            x0[
                (self.n_planets + 1) * 3: (2 * self.n_planets + 1) * 3
            ] = speeds
        return x0

    def mass_from_param(self, param):
        m = self.mass.clone()
        if "mass" in self.what_is_param:
            # I want masses to be positive
            m[: self.n_masses] = torch.sqrt(
                param[: self.n_masses] ** 2 + 1e-64
            )
        return m

    def RK4(self, x, dt, m):
        k1 = self.F(x, m)
        k2 = self.F(x + dt / 2 * k1, m)
        k3 = self.F(x + dt / 2 * k2, m)
        k4 = self.F(x + dt * k3, m)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def F_multi(self, x, m):
        """
        Return [v, f] where f is the gravitational force applied to each planet
                This force is given by the formula f = -G \sum_{i\neq j}

        Parameters
        ----------
        x : tensor of shape (3*(n_planets+1), K)
            concatenate of all planets and sun positions.
            where K is the number of configurations

        m : mass of shape ((n_planets+1), K)
        Returns
        -------
        y : tensor of shape (6*(n_planets+1), K)
            positions and speeds of planets and sun.

        """

        x_init_shape = x.shape

        K = x.shape[-1]

        y = torch.zeros(
            [(self.n_planets + 1) * 6, K],
            device=self.device,
            dtype=self.dtype,
        )

        x_pos = x[: (self.n_planets + 1) * 3, :]
        x_speed = x[(self.n_planets + 1) * 3:, :]

        y[: (self.n_planets + 1) * 3, :] = x_speed

        pos = x_pos.view(((self.n_planets + 1), 3, K))

        pos_diff = pos[:, None, :, :] - pos[None, :, :, :]
        pos_norm = torch.sum(pos_diff**2, dim=2, keepdim=True) ** 1.5 + 1e-16
        masses = m[None, :, None, :]

        intermediate_grad = self.G * masses * pos_diff / pos_norm
        grad = -torch.sum(intermediate_grad, axis=1)

        y[(self.n_planets + 1) * 3:,
          :] = grad.view(y[(self.n_planets + 1) * 3:, :].shape)
        # The sun position and speed are set to 0
        y[-3:, :] *= 0
        return y.view(x_init_shape)

    def F(self, x, m):
        """
        Return [v, f] where f is the gravitational force applied to each planets
                This force is given by the formula f = -G \sum_{i\neq j}

        Parameters
        ----------
        x : tensor of shape 3(n_planets+1)
            concatenate of all planets and sun positions.

        Returns
        -------
        y : tensor of shape 6(n_planets+1)
            positions and speeds of planets and sun.

        """
        y = torch.zeros(
            [(self.n_planets + 1) * 6],
            device=self.device,
            dtype=self.dtype,
        )

        x_pos = x[: (self.n_planets + 1) * 3]
        x_speed = x[(self.n_planets + 1) * 3:]

        y[: (self.n_planets + 1) * 3] = x_speed

        pos = x_pos.view(((self.n_planets + 1), 3))

        pos_diff = pos[:, None, :] - pos[None, :, :]
        pos_norm = torch.sum(pos_diff**2, dim=2, keepdim=True) ** 1.5 + 1e-16
        masses = m[None, :, None]

        intermediate_grad = self.G * masses * pos_diff / pos_norm
        grad = -torch.sum(intermediate_grad, axis=1)

        y[(self.n_planets + 1) * 3:] = grad.ravel()
        # The sun position and speed are set to 0
        y[-3:] *= 0
        return y

    def generate_Phi(self, Tinit, Tfin, dt):
        def Phi(param):
            m = self.mass_from_param(param)
            x0 = self.x_from_param(param)

            output = self.resolution(x0, m, Tinit=Tinit, Tfin=Tfin, dt=dt)
            if self.export_speed:
                return output[:, :: self.subselect_every]
            else:
                return output[: (self.n_planets) * 3, :: self.subselect_every]

        return Phi

    def plot_curves(self, X, lims=30):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for k in range(self.n_planets):
            ax.plot(
                X[3 * k, :].tolist(),
                X[3 * k + 1, :].tolist(),
                X[3 * k + 2, :].tolist(),
            )
        ax.set_xlim([-lims, lims])
        ax.set_ylim([-lims, lims])
        ax.set_zlim([-lims, lims])
        
        plt.legend(self.planet_names)

    def plot_param(self, param):
        fig = plt.figure()

        ax = plt.subplot(2, 2, 1)

        ax.plot(self.mass_from_param(param)[:-1].tolist())
        ax.set_title("masses")

        ax = plt.subplot(2, 2, 2)

        ax.plot(self.x_from_param(param)[: self.n_planets * 3].tolist())
        ax.set_title("positions")

        ax = plt.subplot(2, 2, 3)

        ax.plot(
            self.x_from_param(param)[
                (self.n_planets + 1) * 3: (2 * self.n_planets + 1) * 3
            ].tolist()
        )
        ax.set_title("speeds")

    def plot_inial_planets(self, param, solution=None, with_curves=False):
        x_from_p = self.x_from_param(param)
        m_from_p = self.mass_from_param(param)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for i_planet in range(self.n_planets + 1):
            ax.plot(
                [x_from_p[3 * i_planet].item()],
                [x_from_p[3 * i_planet + 1].item()],
                [x_from_p[3 * i_planet + 2].item()],
                marker="*",
                color=self.planet_colors[i_planet],
                markersize=10 * m_from_p[i_planet].item() ** 0.1,
            )

        if with_curves:
            output = solution.A_op(param)
            for i_planet in range(self.n_planets):
                ax.plot(
                    output[3 * i_planet, :].tolist(),
                    output[3 * i_planet + 1, :].tolist(),
                    output[3 * i_planet + 2, :].tolist(),
                    color=self.planet_colors[i_planet],
                )
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_zlim([-30, 30])



def solar_plot_figures(
    times_list,
    image_directory,
    solution,
    n_pts_per_dim,
    solar_model,
    dt,
    n_planets,
    iter_begin=None,
    iter_end=None,
):
    image_name_base = "solar_system_"

    for i_times, [Tinit, Tfin] in enumerate(times_list):
        # GET RESULTS
        save_name_path = image_directory + f"/expe_{i_times}" + ".pth"
        solution.load_results(save_name_path)

        computed_index = [
            k for k, bo in enumerate(solution.param_is_computed) if bo
        ]
        if iter_begin == None:
            iter_begin = computed_index[0]
        if iter_end == None:
            iter_end = computed_index[-1]

        param = solar_model.param_default

        print(Tinit, Tfin)
        Phi = solar_model.generate_Phi(Tinit, Tfin, dt)

        model = ModelOperator(Phi, param)
        y = model(param)

        suffix = f"Tinit_{Tinit}_Tfin_{Tfin}_n_pts_{n_pts_per_dim}"

        # for k in range(n_pts_per_dim[0]):
        #     if solution.param_is_computed[k]:
        #         print(k)
        #         solar_model.plot_param(solution.param_evals[k, :])
        # plt.suptitle(f"Iteration {k}")
        # plt.show()

        # COMPUTE ALL OUTPUT ALL AT ONCE
        all_outputs = []
        for k in range(0, iter_end + 1):
            print(k)
            if k >= iter_begin:
                all_outputs.append(Phi(solution.param_evals[k, :]))
            else:
                all_outputs.append(None)

        # MASSES
        ratio = 0.5
        plt.figure(figsize=(6 * ratio, 4 * ratio))
        for i_planet in range(n_planets):
            mass_evol = torch.sqrt(
                solution.param_evals[iter_begin: iter_end + 1, i_planet] ** 2
            )
            plt.semilogy(
                mass_evol.tolist(),
                color=colors[i_planet],
                linestyle=linestyles[i_planet],
            )

        plt.xlabel("k")
        plt.ylabel("mass (1 unit = 1 solar mass)")
        plt.legend(solar_model.planet_names)
        save_this_plot(
            image_directory,
            image_name_base + "mass",
            filetype=".png",
            clear=False,
        )
        plt.show()

        # RELATIVE MASSES
        ratio = 0.7
        plt.figure(figsize=(6 * ratio, 4 * ratio))
        mass_init = torch.sqrt(param[:n_planets] ** 2).to("cpu")
        mass_rel = []
        for k in range(iter_begin, iter_end + 1):
            mass = torch.sqrt(solution.param_evals[k, :n_planets] ** 2)
            mass_rel.append(((mass_init - mass).norm() / mass.norm()).item())
            print(mass_init, mass)
        plt.plot(mass_rel, color="b")

        plt.xlabel("k")
        plt.ylabel("relative mass discrepancy")
        save_this_plot(
            image_directory,
            image_name_base + "relative_mass",
            filetype=".png",
            clear=False,
        )
        plt.show()

        # RELATIVE DISCREPANCY INPUT / OUTPUT
        ratio = 0.5
        plt.figure(figsize=(6 * ratio, 4 * ratio))
        mass_init = torch.sqrt(param[:n_planets] ** 2).to("cpu")
        mass_rel = []
        for k in range(iter_begin, iter_end + 1):
            mass = torch.sqrt(solution.param_evals[k, :n_planets] ** 2)
            mass_rel.append(((mass_init - mass).norm() / mass.norm()).item())
        plt.plot(mass_rel, color="b")

        x_evol = []

        for k in range(iter_begin, iter_end + 1):
            output = all_outputs[k]
            x_evol.append(((output - y).norm() / y.norm()).item())
            plt.plot(x_evol, color="g")

        plt.xlabel("k")
        plt.ylabel("relative discrepancy")
        plt.legend(["mass", "trajectory"])
        save_this_plot(
            image_directory,
            image_name_base + "discrepancy_input_output",
            filetype=".png",
            clear=False,
        )
        plt.show()

        # RELATIVE OUTPUT POSITIONS
        ratio = 0.7
        plt.figure(figsize=(6 * ratio, 4 * ratio))

        x_evol = [[] for i in range(n_planets)]

        for k in range(iter_begin, iter_end + 1):
            output = all_outputs[k]
            for i_planet in range(n_planets):
                x_evol[i_planet].append(
                    (
                        (
                            output[3 * i_planet: 3 * i_planet + 3, :]
                            - y[3 * i_planet: 3 * i_planet + 3, :]
                        ).norm()
                        / y[3 * i_planet: 3 * i_planet + 3, :].norm()
                    ).item()
                )
                plt.plot(
                    x_evol[i_planet],
                    color=colors[i_planet],
                    linestyle=linestyles[i_planet],
                )

        plt.xlabel("k")
        plt.ylabel("relative trajectory discrepancy")
        plt.legend(solar_model.planet_names)
        save_this_plot(
            image_directory,
            image_name_base + "relative_output",
            filetype=".png",
            clear=False,
        )
        plt.show()

        # GENERAL RELATIVE OUTPUT POSITIONS
        ratio = 0.7
        plt.figure(figsize=(6 * ratio, 4 * ratio))

        x_evol = []

        for k in range(iter_begin, iter_end + 1):
            output = all_outputs[k]
            x_evol.append(((output - y).norm() / y.norm()).item())
            plt.plot(x_evol, color="b")

        plt.xlabel("k")
        plt.ylabel("relative trajectory discrepancy")
        save_this_plot(
            image_directory,
            image_name_base + "general_relative_output",
            filetype=".png",
            clear=False,
        )
        plt.show()

        # RELATIVE POSITIONS
        ratio = 0.7
        plt.figure(figsize=(6 * ratio, 4 * ratio))
        for i_planet in range(n_planets):
            x_evol = [
                (
                    (
                        solar_model.x_from_param(solution.param_evals[k, :])[
                            3 * i_planet: 3 * i_planet + 3
                        ]
                        - solar_model.x_from_param(param)[
                            3 * i_planet: 3 * i_planet + 3
                        ]
                    ).norm()
                    / (
                        solar_model.x_from_param(param)[
                            3 * i_planet: 3 * i_planet + 3
                        ]
                    ).norm()
                ).item()
                for k in range(iter_begin, iter_end + 1)
            ]
            plt.plot(
                x_evol, color=colors[i_planet], linestyle=linestyles[i_planet]
            )

        plt.xlabel("k")
        plt.ylabel("relative initial position discrepancy")
        plt.legend(solar_model.planet_names)
        save_this_plot(
            image_directory,
            image_name_base + "relative_position",
            filetype=".png",
            clear=False,
        )
        plt.show()

        # RELATIVE SPEEDS
        ratio = 0.7
        plt.figure(figsize=(6 * ratio, 4 * ratio))
        for i_planet in range(n_planets):
            x_evol = [
                (
                    solar_model.x_from_param(solution.param_evals[k, :])[
                        3 * (n_planets + 1)
                        + 3 * i_planet: 3 * (n_planets + 1)
                        + 3 * i_planet
                        + 3
                    ]
                    - solar_model.x_from_param(param)[
                        3 * (n_planets + 1)
                        + 3 * i_planet: 3 * (n_planets + 1)
                        + 3 * i_planet
                        + 3
                    ]
                )
                .norm()
                .item()
                for k in range(iter_begin, iter_end + 1)
            ]
            plt.plot(
                x_evol, color=colors[i_planet], linestyle=linestyles[i_planet]
            )

        plt.xlabel("k")
        plt.ylabel("relative initial speed discrepancy")
        plt.legend(solar_model.planet_names)
        save_this_plot(
            image_directory,
            image_name_base + "relative_speed",
            filetype=".png",
            clear=False,
        )
        plt.show()

        # OUTPUT POSITIONS
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        # For the legend
        for i_planet in range(n_planets):
            ax.plot(
                [1000, 1000],
                [1000, 1001],
                [1000, 1000],
                color=colors[i_planet],
                linewidth=1,
            )

        for k in range(iter_begin, iter_end + 1):
            if solution.param_is_computed[k]:
                X = output = all_outputs[k]
                for i_planet in range(n_planets):
                    ax.plot(
                        X[3 * i_planet, :].tolist(),
                        X[3 * i_planet + 1, :].tolist(),
                        X[3 * i_planet + 2, :].tolist(),
                        color=colors[i_planet],
                        linewidth=0.1,
                    )

        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_zlim([-30, 30])

        plt.legend(solar_model.planet_names)
        save_this_plot(
            image_directory,
            image_name_base + "output_trajectory",
            filetype=".png",
            clear=False,
        )
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Models parameters
    n_planets = 9
    what_is_param = ["mass", "position", "speed"]  # ["mass"]
    Tinit = 10000
    Tfin = 20000
    dt = 1

    image_directory = "./pdf/output_images"
    os.makedirs(image_directory, exist_ok=True)

    solar_model = SolarModel(
        n_planets=n_planets,
        what_is_param=what_is_param,
        solar_mass_included=True,
        image_directory=image_directory,
        device=device,
        dtype=dtype,
    )

    param = solar_model.param_default
    Phi = solar_model.generate_Phi(Tinit, Tfin, dt)

    output = Phi(param)

    lims = max(-output.min().item(), output.max().item())
    solar_model.plot_curves(output, lims=lims)
