def OneDModel(runlength=10000, bpss_reoccurence=1000, bpss_length=500, show_figures=False):
    # ----------------------------------------------------------------------- #
    #   Very rough draft of 1D diffusion/advection model for testing          #
    #   Date: 14/10/2024            
    #   Adapted from 2D diffusion/advection to there are some vestigial
    #   features
    #   Author: Thomas Bauska <thausk@bas.ac.uk>                              #
    #                                                                         #
    #=========================Basic Setup=====================================#
    #-------------------------------------------------------------------------#

    import numpy as np
    import matplotlib.pyplot as plt

    nx = 1     # core boxes akin to longitude (don't change!)
    ny = 4     # core boxes akin to latitude (don't change!)

    #--> NOTE: the boundary condition on the north and south (y) is a Neumann
    # so there are a total of 6 boxes with boxes 1 and 6 forced to equal to boxes 
    # 2 and 5, respectively. They are like ghost boxes such that derivative at the
    # true boundary is zero.

    # Boundary conditions on east and west (x) are wrapped around such that the grid is continuous

    lengthx = 1  # domain length along x (unitless) (don't change!)
    lengthy = 4  # domain length along y (unitless) don't change!)

    dt = 1       # time step [nominally yr]  (there are some stability check below)
    nsteps = int(runlength / dt)  # number of timesteps (feel free to change)

    num_simulations = 10  # Total number of simulations

    # Initialize arrays to save Temp, F, and Fadv for all loops
    Temps = np.zeros((num_simulations, nx, ny + 2, nsteps))
    Fs = np.zeros((num_simulations, nx, ny + 2, nsteps))
    Fadvs = np.zeros((num_simulations, nx, ny + 2, nsteps))
    global_temp_arrs = np.zeros((num_simulations, nsteps))
    area_temp_arrs = np.zeros((num_simulations, nsteps))

    for i in range(1, num_simulations + 1):  # this is a little loop which runs 9 simulations with noise and the last one without.

        # ===============Switches, Knobs and Tunable Parameters=====================#
        # -------------------------------------------------------------------------#
        # --Diffusion--#
        Dy_var = [0.0005, 0.0005, 0.1, 0.1, 0.1]  # Variable diffusive parameter along y  <======CAN CHANGE 
        # wy = 5  # advective flux (currently defunct)
        maxD = 0.1  # some relict code to check numerical stability. Here you can enter maximum diffusion rate
        maxw = 0.0  # some relict code to check numerical stability. Here you can enter maximum advection rate

        # --Noise--#
        if i == 10:
            wnoise_level = 0
        else:
            wnoise_level = 0.005  # total white noise level <======CAN CHANGE 
            # wnoise_level = 0.00005

        wnoise_sigma = np.array([1, 1, 1, 1, 1, 1]) * wnoise_level  # sets level of white noise in each latitude 
        wnoise_dt = 1  # sets timestep for noise (ie. if dt = 1 an annual noise timeseries is interpolated to the model grid)

        # --Response times--#
        arb_bpss_target = 0   # this sets the magnitude of the bipolar seesaw heat flux.
        arb_bpss_tau = 10     # this sets the rate (nominally years) that the model moves towards the full bipolar seesaw heat flux.

        rectifier_global_temp = 0      # this forces the global average temperature to tend to the set value
        rectifier_global_tau = 100     # this sets the rate (nominally years) that the model tends back to the set values. If you want to turn off the effect, set it to a very large number.

        rectifier_area_temp = 0        # this forces a set of grid cells in the model to to the set value.  
        rectifier_area_tau = 100000    # this sets the rate (nominally years) that the model tends back to the set values. If you want to turn off the effect, set it to a very large number.

        gain = [0, 0, 0, 0, 0, 0]      # this currently doesn't work..

        timetrip_pre = np.arange(1, 21)

        timetrip = np.column_stack((
            bpss_reoccurence * timetrip_pre + bpss_length * (timetrip_pre - 1),
            (bpss_reoccurence + bpss_length) * timetrip_pre
        ))

        # ============Pre-processing of user-defined data==========================#
        # -------------------------------------------------------------------------#
        # Calculate grid steps
        hx = lengthx / nx      # grid step along x (unitless) 
        hy = lengthy / ny      # grid step along y (unitless) 

        # Numerical setup: time step (stability conditions)
        sigma = 0.7                       # safety coefficient
        dt_diff = (1 / 4) * min(hx ** 2, hy ** 2) / maxD   # diffusion [yr]
        if maxw == 0:
            dt_conv = np.inf
        else:
            dt_conv = 4 * maxD / (maxw ** 2)  # convection [yr]
        dtstabilitycheck = sigma * min(dt_diff, dt_conv)   # time step [yr]

        if dt > dtstabilitycheck:
            print('might be numerically unstable')
            print(dt / dtstabilitycheck)
        else:
            pass

        # Calculate time steps
        time = np.arange(1, nsteps + 1) * dt

        # Memory allocation
        Temp = np.zeros((nx, ny + 2, nsteps))     # Temperature
        arb_bpss_target_time = np.zeros(nsteps)   # the target value for the bipolar seesaw heat flux parameter. The actual value tends to this in a lagged response.
        arb_bpss_actual_time = np.zeros(nsteps)   # the actual value for the bipolar seesaw heat flux parameter.
        Fadv = np.zeros((nx, ny + 2, nsteps))     # Advective Flux 
        F = np.zeros((nx, ny + 2, nsteps))        # Diffusive Flux
        Fgain = np.zeros((nx, ny + 2, nsteps))    # Gain Flux (not active!)
        Frectifier_global = np.zeros((nx, ny + 2, nsteps))  # Rectifier Flux when using the global mean temperature
        Frectifier_area = np.zeros((nx, ny + 2, nsteps))      # Rectifier Flux when using the just a set area
        wnoiseint = np.zeros((nx, ny + 2, nsteps))           # white noise

        # White noise production
        wnoiserawtime = np.arange(0, time[-1] + wnoise_dt, wnoise_dt)
        wnoiseraw_shape = (nx, ny + 2, len(wnoiserawtime))
        wnoiseraw = wnoise_sigma.reshape(1, -1, 1) * np.random.randn(*wnoiseraw_shape)

        # Interpolate white noise to time grid
        for y in range(ny + 2):
            for x in range(nx):
                wnoiseint[x, y, :] = np.interp(time, wnoiserawtime, wnoiseraw[x, y, :])

        # Initialize arrays for global and area temperature
        global_temp_arr = np.zeros(nsteps)
        area_temp_arr = np.zeros(nsteps)

        # ============================Core Equations===============================#
        # -------------------------------------------------------------------------#
        for t in range(nsteps - 1):
            for y in range(1, 5):
                x = 0
                F[x, y, t + 1] = +(Dy_var[y] * dt / hy ** 2) * (Temp[x, y + 1, t] - Temp[x, y, t]) \
                                + (Dy_var[y - 1] * dt / hy ** 2) * (Temp[x, y - 1, t] - Temp[x, y, t])
            #
            # -----Special Advection Forcing------#
            x = 0
            y = 4
            arb_bpss_target_time[t] = arb_bpss_target
            arb_bpss_actual_time[t + 1] = arb_bpss_actual_time[t] \
                                          + (arb_bpss_target_time[t] - arb_bpss_actual_time[t]) * dt * (1 / arb_bpss_tau)

            Fadv[x, y - 2, t + 1] = -arb_bpss_actual_time[t] * dt
            Fadv[x, y - 1, t + 1] = +0.5 * arb_bpss_actual_time[t] * dt
            Fadv[x, y, t + 1] = +0.5 * arb_bpss_actual_time[t] * dt

            # -----Summing up, Switches and Feedbacks-----#        
            global_temp_arr[t] = Temp[0, 1:5, t].mean()
            area_temp_arr[t] = Temp[0, 2:4, t].mean()
            Frectifier_global[:, :, t + 1] = (rectifier_global_temp - global_temp_arr[t]) * (1 / rectifier_global_tau) * dt

            Frectifier_area[x, 2:4, t + 1] = (rectifier_area_temp - area_temp_arr[t]) * (1 / rectifier_area_tau) * dt
            Temp[:, :, t + 1] = Temp[:, :, t] \
                               + F[:, :, t + 1] \
                               + Fadv[:, :, t + 1] \
                               + wnoiseint[:, :, t + 1] \
                               + Frectifier_global[:, :, t + 1] \
                               + Frectifier_area[:, :, t + 1] \
                               + Fgain[:, :, t]
            Fgain[:, :, t + 1] = np.array(gain).reshape(1, -1) * (Temp[:, :, t + 1] - Temp[:, :, t])

            # ----Finding the right time to turn off and on----#   
            if any((time[t] > timetrip[k, 0]) and (time[t] < timetrip[k, 1]) for k in range(20)):
                arb_bpss_target = 1   # this in ON trip <======CAN CHANGE 
            else:
                arb_bpss_target = -1  # this in OFF trip <======CAN CHANGE 

            # North and South Boundary conditions (Neumann's)
            Temp[0, 0, t + 1] = Temp[0, 1, t + 1]
            Temp[0, 5, t + 1] = Temp[0, 4, t + 1]

        # Save Temp, F, Fadv, global_temp_arr, area_temp_arr for each loop
        idx = i - 1  # Adjust index since Python is 0-based
        Temps[idx] = Temp
        Fs[idx] = F
        Fadvs[idx] = Fadv
        global_temp_arrs[idx] = global_temp_arr
        area_temp_arrs[idx] = area_temp_arr

        # =============================Figures=====================================#
        # -------------------------------------------------------------------------#
        # For Figure 1, plot only for i == 1 or i == 10
        if show_figures and (i == 1 or i == 10):
            plt.figure(1)
            plt.subplot(3, 1, 1)
            plt.plot(time, np.squeeze(np.sum(Temp[:, 1:5, :], axis=(0, 1))), 'ko', markersize=1)
            plt.title(f'Simulation {i}: Sum of Temperatures')
            plt.subplot(3, 1, 2)
            plt.plot(time, np.squeeze(np.sum(F[:, 1:5, :], axis=(0, 1))), 'ko', markersize=1)
            plt.plot(time, np.squeeze(np.sum(Fadv[:, 1:5, :], axis=(0, 1))), 'ro', markersize=1)
            plt.title(f'Simulation {i}: Diffusive and Advective Fluxes')
            plt.subplot(3, 1, 3)
            plt.plot(time[:-1], global_temp_arr[:-1], 'k', linewidth=0.5)
            plt.title(f'Simulation {i}: Global Temperature')
            plt.tight_layout()
            plt.show()

    # After the loop, stack results and plot Figures 2, 3, and 6
    if show_figures:
        # Figure 2: Temperatures of Boxes 1 to 4 over time
        plt.figure(2)
        for idx in range(num_simulations):
            if idx == num_simulations - 1:
                # Last loop (i == 10), plot in black
                plt.plot(time, Temps[idx, 0, 1, :], 'k', linewidth=1)
                plt.plot(time, Temps[idx, 0, 2, :], 'k', linewidth=1)
                plt.plot(time, Temps[idx, 0, 3, :], 'k', linewidth=1)
                plt.plot(time, Temps[idx, 0, 4, :], 'k', linewidth=1)
            else:
                # Other loops, plot in colors with transparency
                plt.plot(time, Temps[idx, 0, 1, :], color='blue', alpha=0.3)
                plt.plot(time, Temps[idx, 0, 2, :], color='cyan', alpha=0.3)
                plt.plot(time, Temps[idx, 0, 3, :], color='red', alpha=0.3)
                plt.plot(time, Temps[idx, 0, 4, :], color='green', alpha=0.3)
                plt.legend(['Box 1', 'Box 2', 'Box 3', 'Box 4'])
        plt.title('Temperatures of Boxes 1 to 4 Over Time')
        plt.xlabel('Time (years)')
        plt.ylabel('Temperature')
        plt.show()

        # Figure 3: Difference between North and South Box Temperatures
        plt.figure(3)
        for idx in range(num_simulations):
            temp_diff = Temps[idx, 0, 4, :] - Temps[idx, 0, 1, :]
            if idx == num_simulations - 1:
                plt.plot(time, temp_diff, 'k', linewidth=1)
            else:
                plt.plot(time, temp_diff, 'k', alpha=0.3)
        plt.title('Temperature Difference Between North and South Boxes')
        plt.xlabel('Time (years)')
        plt.ylabel('Temperature Difference')
        plt.show()

        # Figure 6: Temperatures of South and North Boxes over time
        plt.figure(6)
        for idx in range(num_simulations):
            if idx == num_simulations - 1:
                plt.plot(time, Temps[idx, 0, 1, :], 'k', linewidth=1, label='South Box (No Noise)')
                plt.plot(time, Temps[idx, 0, 4, :], 'k', linewidth=1, label='North Box (No Noise)')
            else:
                plt.plot(time, Temps[idx, 0, 1, :], 'b', alpha=0.3, label='South Box' if idx == 0 else "")
                plt.plot(time, Temps[idx, 0, 4, :], 'g', alpha=0.3, label='North Box' if idx == 0 else "")
        plt.title('Temperatures of South and North Boxes Over Time')
        plt.xlabel('Time (years)')
        plt.ylabel('Temperature')
        plt.legend()
        plt.show()

    return Temps, Fs, Fadvs
