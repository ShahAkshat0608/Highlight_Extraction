import copy
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from SpatialLocalisation.utils import bcolors, find_zero_from_end, find_zero_from_start


class StabilisationManager:
    def __init__(
        self, length, frame_size, stabilise_in_degrees=3, use_constraint_box=True
    ):
        assert length > 1, "Length should be greater than 1"
        self.length = length
        self.stabilise_in_degrees = stabilise_in_degrees
        self.frame_size = frame_size
        self.use_constraint_box = use_constraint_box

    def _convert_to_3_degree(self, rush, length):
        new_rush = []
        for i in range(length):
            x1, y1, x2, y2 = rush[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            height = y2 - y1

            new_rush.append([center_x, center_y, height])

        return new_rush

    def _convert_to_4_degree(self, rush, length):
        new_rush = []
        for i in range(length):
            c_x, c_y, height = rush[i]
            width = height * 9 / 16

            x1 = c_x - width / 2
            x2 = c_x + width / 2
            y1 = c_y - height / 2
            y2 = c_y + height / 2

            new_rush.append([x1, y1, x2, y2])

        return new_rush

    def _solve(self, function, solver, max_iter, verbose, constraints=[]) -> bool:
        # try:
        #     obj = cp.Minimize(function)
        #     prob = cp.Problem(obj, constraints)
        #     prob.solve(verbose=verbose, solver=solver, max_iter=max_iter)
        #     if prob.status != "optimal":
        #         raise Exception("Optimal solution not found with solver", solver)
        #     else:
        #         return
        # except Exception as e:
        #     print("Error: ", e)

        try:
            obj = cp.Minimize(function)
            prob = cp.Problem(obj, constraints)
            prob.solve(verbose=verbose, solver="MOSEK")
            print("Solved with MOSEK with status: ", prob.status)
            if "infeasible" in prob.status:
                raise Exception("Infeasible solution found with solver MOSEK")
            return True
        except Exception as e:
            print("Error: ", e)

        try:
            obj = cp.Minimize(function)
            prob = cp.Problem(obj, constraints)
            prob.solve(verbose=verbose, solver="ECOS")
            print("Solved with ECOS with status: ", prob.status)
            if "infeasible" in prob.status:
                raise Exception("Infeasible solution found with solver ECOS")
            return True
        except Exception as e:
            print("Error: ", e)

        try:
            obj = cp.Minimize(function)
            prob = cp.Problem(obj, constraints)
            prob.solve(verbose=verbose, solver="SCS")
            print("Solved with SCS with status: ", prob.status)
            if "infeasible" in prob.status or "unbounded" in prob.status:
                raise Exception("Infeasible solution found with solver SCS")
            return True
        except Exception as e:
            print("Error: ", e)

        return False

    def _stabilise_in_3_degree(
        self,
        rush,
        constraint_box,
        lambda1,
        lambda2,
        lambday1,  # Not Used
        lambday2,  # Not used
        solver,
        max_iter,
        stabilise_x_y_together,
        frame_height,
        verbose=False,
        plot=False,
        length=None,
    ):
        if plot:
            plt.plot([x[0] for x in rush], label="x_orig")
            plt.plot([x[1] for x in rush], label="y_orig")
            plt.plot([x[2] for x in rush], label="h_orig")
            plt.legend()
            # plt.savefig("graph_original.png")

        if length is None:
            length = self.length

        rush_copy = copy.deepcopy(rush)
        rush_edited_start = False
        rush_edited_end = False

        zero_from_end_index = find_zero_from_end(rush)
        if zero_from_end_index < length:
            length = zero_from_end_index
            rush = rush[:length]
            constraint_box = constraint_box[:length]
            rush_edited_end = True

        zero_from_start_index = find_zero_from_start(rush)
        if zero_from_start_index > 0:
            length = length - zero_from_start_index
            rush = rush[zero_from_start_index:]
            constraint_box = constraint_box[zero_from_start_index:]
            rush_edited_start = True

        rx = cp.Variable(length)
        # ry = cp.Variable(length)
        # rh = cp.Variable(length)
        width = frame_height * 9 / 16

        function_rx = (
            1 / 2 * cp.sum((np.array([x[0] for x in rush]) - rx) ** 2)
            + lambda1 * cp.sum(cp.abs(rx[1:] - rx[:-1]))
            + lambda2 * cp.sum(cp.abs(rx[3:] - 3 * rx[2:-1] + 3 * rx[1:-2] - rx[:-3]))
        )
        # function_ry = (
        #     1 / 2 * cp.sum((np.array([x[1] for x in rush]) - ry) ** 2)
        #     + lambda1 * cp.sum(cp.abs(ry[1:] - ry[:-1]))
        #     + lambda2 * cp.sum(cp.abs(ry[3:] - 3 * ry[2:-1] + 3 * ry[1:-2] - ry[:-3]))
        # )
        # function_rh = (
        #     1 / 2 * cp.sum((np.array([x[2] for x in rush]) - rh) ** 2)
        #     + lambda1 * cp.sum(cp.abs(rh[1:] - rh[:-1]))
        #     + lambda2 * cp.sum(cp.abs(rh[3:] - 3 * rh[2:-1] + 3 * rh[1:-2] - rh[:-3]))
        # )

        # Horizontal constraints
        constraints_h = [rx - width / 2 >= 0, rx + width / 2 <= self.frame_size[1]]
        if self.use_constraint_box:
            constraints_h += [
                rx - width / 2 <= cp.hstack([x[0] for x in constraint_box]),
                cp.hstack([x[2] for x in constraint_box]) <= rx + width / 2,
            ]

        # Vertical constraints
        # constraints_v = [ry - rh / 2 >= 0, ry + rh / 2 <= self.frame_size[0]]
        # if self.use_constraint_box:
        #     print("Using constraint box")
        #     constraints_v += [
        #         ry - rh / 2 <= cp.hstack([x[1] for x in constraint_box]),
        #         cp.hstack([x[3] for x in constraint_box]) <= ry + rh / 2,
        #     ]

        if stabilise_x_y_together:
            solve_return = self._solve(
                (function_rx + function_ry + function_rh),
                solver,
                max_iter,
                verbose,
                (constraints_h + constraints_v),
            )
            if not solve_return:
                print(
                    f"{bcolors.FAIL}Could not solve the problem.\nFalling back to individual x, y, h{bcolors.ENDC}"
                )
                return rush_copy
        else:
            print("Stabilising seperately")
            solve_return1 = self._solve(
                function_rx, solver, max_iter, verbose, constraints_h
            )
            if not solve_return1:
                print(
                    f"{bcolors.FAIL}Could not solve the problem.\nFalling back to individual x{bcolors.ENDC}"
                )
                return rush_copy
            # solve_return2 = self._solve(
            #     function_ry, solver, max_iter, verbose, constraints_v
            # )
            # solve_return3 = self._solve(
            #     function_rh, solver, max_iter, verbose, (constraints_h + constraints_v)
            # )
            # if not (solve_return1 and solve_return2 and solve_return3):
            #     print("Could not solve the problem. Fallback to individual x, y, h")
            #     return rush_copy

        # Rushes are of form x, y, h
        # rush_to_return = [
        #     [rx.value[i], ry.value[i], rh.value[i]] for i in range(length)
        # ]
        # x - center_x, y - center_y, h - height : center_y is fixed : h/2 (as y1=0) where h = frame_height
        rush_to_return = [
            [rx.value[i], frame_height//2, frame_height] for i in range(length)
        ]

        if rush_edited_start:
            rush_to_return = rush_copy[:zero_from_start_index] + rush_to_return

        if rush_edited_end:
            rush_to_return = rush_to_return + rush_copy[zero_from_end_index:]

        if plot:
            plt.plot([x[0] for x in rush_to_return], label="x")
            plt.plot([x[1] for x in rush_to_return], label="y")
            plt.plot([x[2] for x in rush_to_return], label="h")

            if self.use_constraint_box:
                # Plot the constraint box
                plt.plot([x[0] for x in constraint_box], label="x1")
                plt.plot([x[1] for x in constraint_box], label="y1")
                plt.plot([x[2] for x in constraint_box], label="x2")
                plt.plot([x[3] for x in constraint_box], label="y2")

            plt.legend()
            plt.savefig("graph_stable.png")
            plt.clf()

        return rush_to_return

    def _stabilise_in_4_degree(
        self,
        rush,
        constraint_box,
        lambdax1,
        lambdax2,
        lambday1,
        lambday2,
        solver,
        max_iter,
        stabilise_x_y_together,
        verbose=False,
        plot=False,
        length=None,
    ):
        if length is None:
            length = self.length

        rush_copy = copy.deepcopy(rush)
        rush_edited_start = False
        rush_edited_end = False

        zero_from_end_index = find_zero_from_end(rush)
        if zero_from_end_index < length:
            length = zero_from_end_index
            rush = rush[:length]
            constraint_box = constraint_box[:length]
            rush_edited_end = True

        zero_from_start_index = find_zero_from_start(rush)
        if zero_from_start_index > 0:
            length = length - zero_from_start_index
            rush = rush[zero_from_start_index:]
            constraint_box = constraint_box[zero_from_start_index:]
            rush_edited_start = True

        rx1 = cp.Variable(length)
        ry1 = cp.Variable(length)
        rx2 = cp.Variable(length)
        ry2 = cp.Variable(length)

        function_rx1 = (
            1 / 2 * cp.sum((np.array([x[0] for x in rush]) - rx1) ** 2)
            + lambdax1 * cp.sum(cp.abs(rx1[1:] - rx1[:-1]))
            + lambdax2
            * cp.sum(cp.abs(rx1[3:] - 3 * rx1[2:-1] + 3 * rx1[1:-2] - rx1[:-3]))
        )
        function_ry1 = (
            1 / 2 * cp.sum((np.array([x[1] for x in rush]) - ry1) ** 2)
            + lambday1 * cp.sum(cp.abs(ry1[1:] - ry1[:-1]))
            + lambday2
            * cp.sum(cp.abs(ry1[3:] - 3 * ry1[2:-1] + 3 * ry1[1:-2] - ry1[:-3]))
        )
        function_rx2 = (
            1 / 2 * cp.sum((np.array([x[2] for x in rush]) - rx2) ** 2)
            + lambdax1 * cp.sum(cp.abs(rx2[1:] - rx2[:-1]))
            + lambdax2
            * cp.sum(cp.abs(rx2[3:] - 3 * rx2[2:-1] + 3 * rx2[1:-2] - rx2[:-3]))
        )
        function_ry2 = (
            1 / 2 * cp.sum((np.array([x[3] for x in rush]) - ry2) ** 2)
            + lambday1 * cp.sum(cp.abs(ry2[1:] - ry2[:-1]))
            + lambday2
            * cp.sum(cp.abs(ry2[3:] - 3 * ry2[2:-1] + 3 * ry2[1:-2] - ry2[:-3]))
        )

        # Horizontal constraints
        constraints_h = [
            rx1 <= cp.hstack([x[0] for x in rush]),  # x1 of rush >= rx1 (top-left x)
            rx2
            >= cp.hstack([x[2] for x in rush]),  # x2 of rush <= rx2 (bottom-right x)
            # rx1
            # <= cp.hstack([x[0] for x in constraint_box]),  # x1 of constraint_box >= rx1
            # rx2
            # >= cp.hstack([x[2] for x in constraint_box]),  # x2 of constraint_box <= rx2
            rx2 - rx1 == 480 * 9 / 16,
        ]
        constraints_h += [
            rx1 >= 0,
            rx1 <= self.frame_size[1],
            rx2 >= 0,
            rx2 <= self.frame_size[1],
        ]

        # Vertical constraints
        constraints_v = [
            ry1 <= cp.hstack([x[1] for x in rush]),  # y1 of rush >= ry1 (top-left y)
            ry2
            >= cp.hstack([x[3] for x in rush]),  # y2 of rush <= ry2 (bottom-right y)
            # ry1
            # <= cp.hstack([x[1] for x in constraint_box]),  # y1 of constraint_box >= ry1
            # ry2
            # >= cp.hstack([x[3] for x in constraint_box]),  # y2 of constraint_box <= ry2
        ]
        constraints_v += [
            ry1 >= 0,
            ry1 <= self.frame_size[0],
            ry2 >= 0,
            ry2 <= self.frame_size[0],
        ]

        if stabilise_x_y_together:
            solve_return = self._solve(
                (function_rx1 + function_rx2 + function_ry1 + function_ry2),
                solver,
                max_iter,
                verbose,
                (constraints_h + constraints_v),
            )
            if not solve_return:
                print(
                    "Could not solve the problem. Fallback to individual x1, y1, x2, y2"
                )
                return rush_copy
        else:
            solve_return1 = self._solve(
                (function_rx1 + function_rx2), solver, max_iter, verbose, constraints_h
            )
            solve_return2 = self._solve(
                (function_ry1 + function_ry2), solver, max_iter, verbose, constraints_v
            )
            if not (solve_return1 and solve_return2):
                print(
                    "Could not solve the problem. Fallback to individual x1, y1, x2, y2"
                )
                return rush_copy

        # Rushes are of form x1, y1, x2, y2
        rush_to_return = [
            [rx1.value[i], ry1.value[i], rx2.value[i], ry2.value[i]]
            for i in range(length)
        ]

        if rush_edited_start:
            rush_to_return = rush_copy[:zero_from_start_index] + rush_to_return

        if rush_edited_end:
            rush_to_return = rush_to_return + rush_copy[zero_from_end_index:]

        if plot:
            plt.plot([x[0] for x in rush], label="x1")
            plt.plot([x[1] for x in rush], label="y1")
            plt.plot([x[2] for x in rush], label="x2")
            plt.plot([x[3] for x in rush], label="y2")

            plt.plot([x[0] for x in rush_to_return], label="x1_stable")
            plt.plot([x[1] for x in rush_to_return], label="y1_stable")
            plt.plot([x[2] for x in rush_to_return], label="x2_stable")
            plt.plot([x[3] for x in rush_to_return], label="y2_stable")
            plt.legend()
            plt.savefig("graph_with_4.png")

        return rush_to_return

    def stabilise_rushes(
        self,
        rush,
        constraint_box,
        lambdax1,
        lambdax2,
        lambday1,
        lambday2,
        frame_height,
        solver="OSQP",
        max_iter=10000,
        stabilise_x_y_together=True,
        verbose=False,
        plot=True,
    ):
        for index, box in enumerate(constraint_box):
            assert (
                box[0] >= 0
                and box[1] >= 0
                and box[2] <= self.frame_size[1]
                and box[3] <= self.frame_size[0]
            ), "Constraint box should be within frame size for index " + str(index)

        # start_zero = 0
        # end_zero = self.length
        length = self.length
        rush_copy = rush.copy()
        constraint_box_copy = constraint_box.copy()
        # if np.sum(rush[0]) == 0:
        #     # Find till how many index rush is zero
        #     for i, r in enumerate(rush):
        #         if np.sum(r) != 0:
        #             start_zero = i
        #             length = length - start_zero
        #             break

        # if np.sum(rush[-1]) == 0:
        #     # Find till how many index rush is zero
        #     for i, r in enumerate(rush[::-1]):
        #         if np.sum(r) != 0:
        #             end_zero = i + 1
        #             length = length - (self.length - end_zero)
        #             break

        # if start_zero != 0 or end_zero != self.length:
        #     rush_copy = rush_copy[start_zero:end_zero]
        #     constraint_box_copy = constraint_box_copy[start_zero:end_zero]

        if self.stabilise_in_degrees == 3:
            rush_to_stabilise = self._convert_to_3_degree(rush_copy, length)
            return_rush = self._stabilise_in_3_degree(
                rush_to_stabilise,
                constraint_box_copy,
                lambdax1,
                lambdax2,
                lambday1,
                lambday2,
                solver,
                max_iter,
                stabilise_x_y_together,
                frame_height,
                verbose,
                plot=plot,
                length=length,
            )

            # if start_zero != 0 or end_zero != self.length:
            #     return_rush = self._convert_to_4_degree(return_rush, length)
            #     rush[start_zero:end_zero] = return_rush
            #     return rush
            return self._convert_to_4_degree(return_rush, self.length)
        elif self.stabilise_in_degrees == 4:
            return_rush = self._stabilise_in_4_degree(
                rush_copy,
                constraint_box_copy,
                lambdax1,
                lambdax2,
                lambday1,
                lambday2,
                solver,
                max_iter,
                stabilise_x_y_together,
                verbose,
                plot=plot,
                length=length,
            )

            # if start_zero != 0 or end_zero != self.length:
            #     rush[start_zero:end_zero] = return_rush
            #     return rush

            return return_rush
        else:
            return None