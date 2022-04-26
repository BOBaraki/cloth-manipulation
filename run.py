#!/usr/bin/env python
"""
Run before using this file:
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so


This script has been copied from datagen_sideways_fold.py, consult that for changes
"""
import os
from gym import envs
import argparse
import numpy as np
import random
from randomizer.wrappers import RandomizedEnvWrapper
import pdb

import csv

DIR = "/home/obarbany/Data/edo_sample/"

render_mode = "human"

header = ["filename", "cloth_state", "gripper1_state", "gripper2_state", "action"]

data = []


def rescale_action(action, speed, noise):
    noise = np.random.normal(0, noise, len(action))
    action = [x * 60 for x in action]
    actionRescaled = np.clip(action + noise, -speed, speed).tolist()

    return actionRescaled


def generate_demos(obs, render, max_episode_steps, behavior):
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    timeStep = 0
    reach_threshold = 0.03  # how close to get to reach a point in space
    drop_threshold = 0.15
    lowering_threshold = 1.0
    lowering_threshold_one_hand = 0.1
    noise_param = 0.6

    episode_flagLifted = False

    episodeObs.append(obs)

    pick_up_object = 1
    pick_up_object_2 = 0
    place_pos = 2
    ini_obs = obs.copy()
    init_objectPos = np.array(
        [
            ini_obs["observation"][7:10].copy(),
            ini_obs["observation"][10:13].copy(),
            ini_obs["observation"][13:16].copy(),
            ini_obs["observation"][16:19].copy(),
        ]
    )

    initial_dist = np.linalg.norm(init_objectPos[0] - init_objectPos[3])

    while True:
        print("Approaching air", timeStep)
        if render:
            env.render(mode=render_mode)
        obsDataNew = obs.copy()

        # The obsDataNew will change because it will return the second gripper
        objectPos = np.array(
            [
                obsDataNew["observation"][7:10].copy(),
                obsDataNew["observation"][10:13].copy(),
                obsDataNew["observation"][13:16].copy(),
                obsDataNew["observation"][16:19].copy(),
            ]
        )

        # pdb.set_trace()

        # New gripperPos and gripperState
        gripperPos = obsDataNew["observation"][:3].copy()
        gripperState = obsDataNew["observation"][3]
        print(gripperPos)
        print(gripperState)

        gripperPos_2 = obsDataNew["observation"][19:22].copy()
        gripperState_2 = obsDataNew["observation"][22]

        # Save gripper positions and states
        if not os.path.isdir(os.path.join(DIR, "gripper_1")):
            os.mkdir(os.path.join(DIR, "gripper_1"))
        if not os.path.isdir(os.path.join(DIR, "gripper_2")):
            os.mkdir(os.path.join(DIR, "gripper_2"))

        np.savetxt(os.path.join(DIR, "gripper_1", f"pos_{timeStep}.txt"), gripperPos)
        np.savetxt(
            os.path.join(DIR, "gripper_1", f"state_{timeStep}.txt"), [gripperState]
        )

        np.savetxt(os.path.join(DIR, "gripper_2", f"pos_{timeStep}.txt"), gripperPos_2)
        np.savetxt(
            os.path.join(DIR, "gripper_2", f"state_{timeStep}.txt"), [gripperState_2]
        )

        # pdb.set_trace()
        # New object positions related to the gripper maybe add some waiting actions
        # if one gripper grabs the cloth before the other
        object_rel_pos = objectPos - gripperPos
        object_oriented_goal = object_rel_pos[pick_up_object].copy()

        object_rel_pos_2 = objectPos - gripperPos_2
        object_oriented_goal_2 = object_rel_pos_2[pick_up_object_2].copy()
        # pdb.set_trace()
        object_oriented_goal[2] += 0.02
        object_oriented_goal_2[2] += 0.02
        if (
            behavior == "one-hand"
            or behavior == "diagonally"
            or behavior == "onehand-lifting"
            or behavior == "onehand-dropping"
            or behavior == "onehand-lowering"
        ):
            if (
                np.linalg.norm(object_oriented_goal)
            ) <= reach_threshold or timeStep >= max_episode_steps:
                break
        elif (
            np.linalg.norm(object_oriented_goal) <= reach_threshold
            and np.linalg.norm(object_oriented_goal_2) <= reach_threshold
        ) or timeStep >= max_episode_steps:
            break

        # increase the actions space here for two agents
        action = [
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
        ]
        speed = 1.0  # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]

        for i in range(len(object_oriented_goal_2)):
            if (
                behavior == "one-hand"
                or behavior == "diagonally"
                or behavior == "onehand-lifting"
                or behavior == "onehand-dropping"
                or behavior == "onehand-lowering"
            ):
                action[4 + i] = 0.0
            else:
                action[4 + i] = object_oriented_goal_2[i]

        actionRescaled = rescale_action(action, speed, 0.1)

        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1
        obsDataNew = obs.copy()
        obsFilename = obsDataNew["filename"]

        newData = [obsFilename, "flat", "free", "free", "approaching"]

        data.append(newData)
    while True:
        if render:
            env.render(mode=render_mode)
        obsDataNew = obs.copy()
        objectPos = np.array(
            [
                obsDataNew["observation"][7:10].copy(),
                obsDataNew["observation"][10:13].copy(),
                obsDataNew["observation"][13:16].copy(),
                obsDataNew["observation"][16:19].copy(),
            ]
        )
        gripperPos = obsDataNew["observation"][:3].copy()
        gripperState = obsDataNew["observation"][3]

        object_rel_pos = objectPos - gripperPos

        gripperPos_2 = obsDataNew["observation"][19:22].copy()
        gripperState_2 = obsDataNew["observation"][22]

        object_oriented_goal = object_rel_pos[pick_up_object].copy()
        object_oriented_goal[0] += 0.02
        object_oriented_goal_2[0] += 0.02

        object_rel_pos_2 = objectPos - gripperPos_2
        object_oriented_goal_2 = object_rel_pos_2[pick_up_object_2].copy()

        if (
            behavior == "one-hand"
            or behavior == "diagonally"
            or behavior == "onehand-lifting"
            or behavior == "onehand-dropping"
            or behavior == "onehand-lowering"
        ):
            if (
                np.linalg.norm(object_oriented_goal)
            ) <= reach_threshold or timeStep >= max_episode_steps:
                break
        elif (
            np.linalg.norm(object_oriented_goal) <= reach_threshold
            and np.linalg.norm(object_oriented_goal_2) <= reach_threshold
        ) or timeStep >= max_episode_steps:
            break

        action = [
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
        ]
        speed = 1.0  # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]

        for i in range(len(object_oriented_goal_2)):
            if (
                behavior == "one-hand"
                or behavior == "diagonally"
                or behavior == "onehand-lifting"
                or behavior == "onehand-dropping"
                or behavior == "onehand-lowering"
            ):
                action[4 + i] = 0.0
            else:
                action[4 + i] = object_oriented_goal_2[i]

        actionRescaled = rescale_action(action, speed, 0.1)

        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        obsDataNew = obs.copy()
        obsFilename = obsDataNew["filename"]

        newData = [obsFilename, "flat", "free", "free", "approaching"]

        data.append(newData)

        timeStep += 1

    speed = 1.0
    if render:
        env.render(mode=render_mode)
    action = [
        random.uniform(-0.00001, 0.00001),
        random.uniform(-0.00001, 0.00001),
        random.uniform(-0.00001, 0.00001),
        random.uniform(0.6, 1.0),
        random.uniform(-0.00001, 0.00001),
        random.uniform(-0.00001, 0.00001),
        random.uniform(-0.00001, 0.00001),
        random.uniform(0.6, 1.0),
    ]

    actionRescaled = rescale_action(action, speed, 0.1)
    # pdb.set_trace()
    obs, reward, done, info = env.step(actionRescaled)
    obsDataNew = obs.copy()
    obsFilename = obsDataNew["filename"]
    # in case of middle change the label here
    newData = [
        obsFilename,
        "semi-lifted-twohands-middle",
        "closed",
        "closed",
        "picking",
    ]

    if (
        behavior == "one-hand"
        or behavior == "diagonally"
        or behavior == "onehand-lifting"
        or behavior == "onehand-dropping"
        or behavior == "onehand-lowering"
    ):
        newData = [obsFilename, "semi-lifted-onehand", "free", "closed", "picking"]

    data.append(newData)

    episodeAcs.append(actionRescaled)
    episodeObs.append(obs)
    episodeInfo.append(info)

    timeStep += 1
    while True:
        if render:
            env.render(mode=render_mode)
        obsDataNew = obs.copy()
        objectPos = np.array(
            [
                obsDataNew["observation"][7:10].copy(),
                obsDataNew["observation"][10:13].copy(),
                obsDataNew["observation"][13:16].copy(),
                obsDataNew["observation"][16:19].copy(),
            ]
        )
        gripperPos = obsDataNew["observation"][:3].copy()
        gripperState = obsDataNew["observation"][3]

        object_rel_pos = objectPos - gripperPos

        gripperPos_2 = obsDataNew["observation"][19:22].copy()
        gripperState_2 = obsDataNew["observation"][22]

        if behavior == "diagonally":
            object_oriented_goal = obsDataNew["desired_goal"].copy() - gripperPos
        else:
            object_oriented_goal = obsDataNew["desired_goal"].copy()[3:6] - gripperPos

        if (
            behavior != "lifting"
            or behavior != "lowering"
            or behavior != "dropping"
            or behavior != "onahand-lifting"
            or behavior == "complex"
        ):
            object_oriented_goal[1] -= 0.09
        object_oriented_goal[2] += 0.1

        object_oriented_goal_2 = obsDataNew["desired_goal"].copy()[0:3] - gripperPos_2

        object_oriented_goal_2[2] += 0.1

        if behavior == "sideways":
            object_oriented_goal_2[1] -= 0.09

        if behavior == "lowering" and timeStep > 70:
            if (
                np.linalg.norm(object_oriented_goal) <= lowering_threshold
                and np.linalg.norm(object_oriented_goal_2) <= lowering_threshold
            ) or timeStep >= max_episode_steps:
                break
        elif behavior == "dropping" or behavior == "lifting" or behavior == "complex":
            if (
                np.linalg.norm(object_oriented_goal) <= drop_threshold
                and np.linalg.norm(object_oriented_goal_2) <= drop_threshold
            ) or timeStep >= max_episode_steps:
                break
        elif (
            behavior == "one-hand"
            or behavior == "diagonally"
            or behavior == "onehand-lifting"
        ):
            if (
                np.linalg.norm(object_oriented_goal)
            ) <= reach_threshold or timeStep >= max_episode_steps:
                break
        elif behavior == "onehand-dropping":
            if (
                np.linalg.norm(object_oriented_goal) <= drop_threshold
                or timeStep >= max_episode_steps
            ):
                break
        elif behavior == "onehand-lowering":
            if (
                np.linalg.norm(object_oriented_goal) <= lowering_threshold_one_hand
                or timeStep >= max_episode_steps
            ):
                break
        else:
            if (
                np.linalg.norm(object_oriented_goal) <= reach_threshold
                and np.linalg.norm(object_oriented_goal_2) <= reach_threshold
            ) or timeStep >= max_episode_steps:
                break

        action = [
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(0.6, 1.0),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(0.6, 1.0),
        ]

        if place_pos == 3 or behavior == "onehand-lifting":
            speed = 0.356
        elif behavior == "onehand-dropping":
            speed = 0.336
        else:
            speed = 0.556 + np.random.uniform(
                -0.3, 0.3
            )  # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]

        for i in range(len(object_oriented_goal_2)):
            if (
                behavior == "one-hand"
                or behavior == "diagonally"
                or behavior == "onehand-lifting"
                or behavior == "onehand-dropping"
                or behavior == "onehand-lowering"
            ):
                action[4 + i] = 0.0
            else:
                action[4 + i] = object_oriented_goal_2[i]

        actionRescaled = rescale_action(action, speed, noise_param)
        actionRescaled[3] = action[3]
        actionRescaled[7] = action[7]
        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        obsDataNew = obs.copy()

        obsFilename = obsDataNew["filename"]

        temp_obs = np.array(
            [
                obsDataNew["observation"][7:10].copy(),
                obsDataNew["observation"][10:13].copy(),
                obsDataNew["observation"][13:16].copy(),
                obsDataNew["observation"][16:19].copy(),
            ]
        )

        # in case of middle change the label here
        newData = [
            obsFilename,
            "semi-lifted-twohands-middle",
            "closed",
            "closed",
            "picking",
        ]

        if (
            behavior == "one-hand"
            or behavior == "diagonally"
            or behavior == "onehand-lifting"
            or behavior == "onehand-dropping"
            or behavior == "onehand-lowering"
        ):
            newData = [obsFilename, "semi-lifted-onehand", "free", "closed", "picking"]
        liftedFlag = all(i > 0.012 for i in (temp_obs[:, 2] - init_objectPos[:, 2]))

        if liftedFlag:
            episode_flagLifted = True

        # semi-lifted-crambled condition for one hand lifting should go here. same with lifted
        crambled_flag = (
            np.linalg.norm(temp_obs[0] - temp_obs[3]) < initial_dist * 95 / 100
        )
        if liftedFlag:
            if (
                behavior == "onehand-lifting"
                or behavior == "onehand-dropping"
                or behavior == "onehand-lowering"
            ):
                newData = [
                    obsFilename,
                    "lifted-onehand",
                    "free",
                    "closed",
                    "lifting_onehand",
                ]
            else:
                newData = [
                    obsFilename,
                    "lifted-twohands",
                    "closed",
                    "closed",
                    "lifting_two_hands",
                ]
        elif crambled_flag:
            if (
                behavior == behavior == "onehand-lifting"
                or behavior == "onehand-dropping"
                or behavior == "onehand-lowering"
            ):
                newData = [
                    obsFilename,
                    "semi-lifted-crambled",
                    "free",
                    "closed",
                    "lifting_onehand",
                ]

        data.append(newData)

        timeStep += 1
    if behavior == "dropping" or behavior == "onehand-dropping":
        while (timeStep) < max_episode_steps:
            if render:
                env.render(mode=render_mode)
            actionDull = [
                random.uniform(-0.00000001, 0.00000001),
                random.uniform(-0.00000001, 0.00000001),
                random.uniform(0.0, 1.0),
                random.uniform(-0.00000001, 0.00000001),
                random.uniform(-0.00000001, 0.00000001),
                random.uniform(-0.00000001, 0.00000001),
                random.uniform(0.0, 1.0),
                random.uniform(-0.00000001, 0.00000001),
            ]
            actionRescaled = rescale_action(actionDull, 1.0, 0.0)
            obs, reward, done, info = env.step(actionRescaled)

            obsDataNew = obs.copy()
            obsFilename = obsDataNew["filename"]

            newData = [obsFilename, "crampled", "free", "free", "waiting-crampled"]

            if behavior == "one-hand" or behavior == "diagonally":
                newData = [obsFilename, "semi-lifted", "free", "closed", "picking"]

            data.append(newData)

            episodeAcs.append(actionRescaled)
            episodeObs.append(obs)
            episodeInfo.append(info)
            timeStep += 1
    elif (
        behavior == "lowering"
        or behavior == "onehand-lowering"
        or behavior == "lifting"
    ):
        if behavior != "lifting":
            while True:
                if render:
                    env.render(mode=render_mode)
                obsDataNew = obs.copy()
                objectPos = np.array(
                    [
                        obsDataNew["observation"][7:10].copy(),
                        obsDataNew["observation"][10:13].copy(),
                        obsDataNew["observation"][13:16].copy(),
                        obsDataNew["observation"][16:19].copy(),
                    ]
                )
                gripperPos = obsDataNew["observation"][:3].copy()

                gripperState = obsDataNew["observation"][3]

                object_rel_pos = objectPos - gripperPos

                gripperPos_2 = obsDataNew["observation"][19:22].copy()
                gripperState_2 = obsDataNew["observation"][22]

                object_oriented_goal = (
                    obsDataNew["desired_goal"].copy()[3:6] - [0, 0, 0.2] - gripperPos
                )

                object_oriented_goal[2] -= 0.1

                object_oriented_goal_2 = (
                    obsDataNew["desired_goal"].copy()[0:3] - [0, 0, 0.2] - gripperPos_2
                )

                object_oriented_goal_2[2] -= 0.1

                if behavior == "lowering":
                    if (
                        np.linalg.norm(object_oriented_goal) <= reach_threshold
                        and np.linalg.norm(object_oriented_goal_2) <= reach_threshold
                    ) or timeStep >= max_episode_steps:
                        break
                elif behavior == "onehand-lowering":
                    if (
                        np.linalg.norm(object_oriented_goal) <= reach_threshold
                        or timeStep >= max_episode_steps
                    ):
                        np.linalg.norm(object_oriented_goal)
                        break

                action = [
                    random.uniform(-0.00001, 0.00001),
                    random.uniform(-0.00001, 0.00001),
                    random.uniform(-0.00001, 0.00001),
                    random.uniform(0.6, 1.0),
                    random.uniform(-0.00001, 0.00001),
                    random.uniform(-0.00001, 0.00001),
                    random.uniform(-0.00001, 0.00001),
                    random.uniform(0.6, 1.0),
                ]

                if place_pos == 3:
                    speed = 0.356
                else:
                    speed = 0.256  # cap action to whatever speed you want
                speed = 0.556 + np.random.uniform(-0.3, 0.3)
                for i in range(len(object_oriented_goal)):
                    action[i] = object_oriented_goal[i]

                for i in range(len(object_oriented_goal_2)):
                    if behavior == "onehand-lowering":
                        action[4 + i] = 0.0
                    else:
                        action[4 + i] = object_oriented_goal_2[i]

                actionRescaled = rescale_action(action, speed, noise_param)
                actionRescaled[3] = action[3]
                actionRescaled[7] = action[7]
                obs, reward, done, info = env.step(actionRescaled)
                obsDataNew = obs.copy()
                obsFilename = obsDataNew["filename"]
                temp_obs = np.array(
                    [
                        obsDataNew["observation"][7:10].copy(),
                        obsDataNew["observation"][10:13].copy(),
                        obsDataNew["observation"][13:16].copy(),
                        obsDataNew["observation"][16:19].copy(),
                    ]
                )
                newData = [
                    obsFilename,
                    "semi-lifted-twohands",
                    "closed",
                    "closed",
                    "picking",
                ]

                if (
                    behavior == "one-hand"
                    or behavior == "diagonally"
                    or behavior == "onehand-lowering"
                    or behavior == "onehand-lifting"
                ):
                    newData = [
                        obsFilename,
                        "semi-lifted-onehand",
                        "free",
                        "closed",
                        "picking",
                    ]

                liftedFlag = all(
                    i > 0.015 for i in (temp_obs[:, 2] - init_objectPos[:, 2])
                )
                if liftedFlag:
                    episode_flagLifted = True

                if liftedFlag:
                    if (
                        behavior == "onehand-lifting"
                        or behavior == "onehand-dropping"
                        or behavior == "onehand-lowering"
                    ):
                        newData = [
                            obsFilename,
                            "lifted-onehand",
                            "free",
                            "closed",
                            "lifting_onehand",
                        ]
                    else:
                        newData = [
                            obsFilename,
                            "lifted",
                            "closed",
                            "closed",
                            "lifting_two_hands",
                        ]
                else:
                    if behavior == "onehand-lowering" and episode_flagLifted:
                        newData = [
                            obsFilename,
                            "semi-lifted-crampled",
                            "free",
                            "closed",
                            "lowering-onehand",
                        ]

                data.append(newData)
                episodeAcs.append(actionRescaled)
                episodeObs.append(obs)
                episodeInfo.append(info)

                timeStep += 1

        while (timeStep) < max_episode_steps:
            if render:
                env.render(mode=render_mode)
            actionDull = [
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(0.6, 1.0),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(0.6, 1.0),
            ]
            actionRescaled = rescale_action(actionDull, 1.0, 0.0)
            obs, reward, done, info = env.step(actionRescaled)
            obsDataNew = obs.copy()
            obsFilename = obsDataNew["filename"]
            # in case of middle change the label here
            newData = [
                obsFilename,
                "semi-lifted-twohands-middle",
                "closed",
                "closed",
                "picking",
            ]

            if (
                behavior == "one-hand"
                or behavior == "diagonally"
                or behavior == "onehand-lowering"
                or behavior == "onehand-lifting"
            ):
                newData = [
                    obsFilename,
                    "semi-lifted-onehand",
                    "free",
                    "closed",
                    "picking",
                ]

            liftedFlag = all(i > 0.01 for i in (temp_obs[:, 2] - init_objectPos[:, 2]))
            if liftedFlag:
                episode_flagLifted = True

            if liftedFlag:
                if (
                    behavior == "onehand-lifting"
                    or behavior == "onehand-dropping"
                    or behavior == "onehand-lowering"
                ):
                    newData = [
                        obsFilename,
                        "lifted-onehand2",
                        "free",
                        "closed",
                        "lifting_onehand",
                    ]
                else:
                    newData = [
                        obsFilename,
                        "lifted-twohands",
                        "closed",
                        "closed",
                        "lifting_two_hands",
                    ]
            else:
                if behavior == "onehand-lowering" and episode_flagLifted:
                    newData = [
                        obsFilename,
                        "semi-lifted-crampled",
                        "free",
                        "closed",
                        "lowering-onehand",
                    ]

            data.append(newData)
            episodeAcs.append(actionRescaled)
            episodeObs.append(obs)
            episodeInfo.append(info)
            timeStep += 1
    elif behavior == "complex":
        while True:
            if render:
                env.render(mode=render_mode)
            obsDataNew = obs.copy()
            objectPos = np.array(
                [
                    obsDataNew["observation"][7:10].copy(),
                    obsDataNew["observation"][10:13].copy(),
                    obsDataNew["observation"][13:16].copy(),
                    obsDataNew["observation"][16:19].copy(),
                ]
            )
            gripperPos = obsDataNew["observation"][:3].copy()
            gripperState = obsDataNew["observation"][3]

            object_rel_pos = objectPos - gripperPos

            gripperPos_2 = obsDataNew["observation"][19:22].copy()
            gripperState_2 = obsDataNew["observation"][22]

            object_oriented_goal = (
                obsDataNew["desired_goal"].copy()[3:6] - [0, 0.3, 0.0] - gripperPos
            )

            object_oriented_goal[1] -= 0.1

            object_oriented_goal_2 = (
                obsDataNew["desired_goal"].copy()[0:3] - [0, 0.3, 0.0] - gripperPos_2
            )

            object_oriented_goal_2[1] -= 0.1

            if behavior == "complex":
                if (
                    np.linalg.norm(object_oriented_goal) <= drop_threshold
                    and np.linalg.norm(object_oriented_goal_2) <= drop_threshold
                ) or timeStep >= max_episode_steps:
                    break
            action = [
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(0.6, 1.0),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(0.6, 1.0),
            ]

            if place_pos == 3:
                speed = 0.356
            else:
                speed = 0.856  # cap action to whatever speed you want
            speed = 0.556 + np.random.uniform(-0.3, 0.3)
            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]

            for i in range(len(object_oriented_goal_2)):
                if behavior == "onehand-lowering":
                    action[4 + i] = 0.0
                else:
                    action[4 + i] = object_oriented_goal_2[i]

            actionRescaled = rescale_action(action, speed, noise_param)
            actionRescaled[3] = action[3]
            actionRescaled[7] = action[7]
            obs, reward, done, info = env.step(actionRescaled)
            obsDataNew = obs.copy()
            obsFilename = obsDataNew["filename"]

            newData = [
                obsFilename,
                "semi-lifted-twohands",
                "closed",
                "closed",
                "picking",
            ]

            if (
                behavior == "one-hand"
                or behavior == "diagonally"
                or behavior == "onehand-dropping"
            ):
                newData = [
                    obsFilename,
                    "semi-lifted-onehand",
                    "free",
                    "closed",
                    "picking",
                ]

            data.append(newData)

            episodeAcs.append(actionRescaled)
            episodeObs.append(obs)
            episodeInfo.append(info)

            timeStep += 1

        while True:
            if render:
                env.render(mode=render_mode)
            obsDataNew = obs.copy()
            objectPos = np.array(
                [
                    obsDataNew["observation"][7:10].copy(),
                    obsDataNew["observation"][10:13].copy(),
                    obsDataNew["observation"][13:16].copy(),
                    obsDataNew["observation"][16:19].copy(),
                ]
            )
            gripperPos = obsDataNew["observation"][:3].copy()
            gripperState = obsDataNew["observation"][3]

            object_rel_pos = objectPos - gripperPos

            gripperPos_2 = obsDataNew["observation"][19:22].copy()
            gripperState_2 = obsDataNew["observation"][22]

            object_oriented_goal = (
                obsDataNew["desired_goal"].copy()[3:6] - [0, 0.4, 0.2] - gripperPos
            )

            object_oriented_goal[2] -= 0.1

            object_oriented_goal_2 = (
                obsDataNew["desired_goal"].copy()[0:3] - [0, 0.4, 0.2] - gripperPos_2
            )

            object_oriented_goal_2[2] -= 0.1

            if behavior == "complex":
                if (
                    np.linalg.norm(object_oriented_goal) <= drop_threshold
                    and np.linalg.norm(object_oriented_goal_2) <= drop_threshold
                ) or timeStep >= max_episode_steps:
                    break

            action = [
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(0.6, 1.0),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(0.6, 1.0),
            ]

            if place_pos == 3:
                speed = 0.356
            else:
                speed = 0.856  # cap action to whatever speed you want
            speed = 0.556 + np.random.uniform(-0.3, 0.3)

            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]

            for i in range(len(object_oriented_goal_2)):
                if behavior == "onehand-lowering":
                    action[4 + i] = 0.0
                else:
                    action[4 + i] = object_oriented_goal_2[i]

            actionRescaled = rescale_action(action, speed, noise_param)
            actionRescaled[3] = action[3]
            actionRescaled[7] = action[7]
            obs, reward, done, info = env.step(actionRescaled)
            obsDataNew = obs.copy()
            obsFilename = obsDataNew["filename"]

            newData = [
                obsFilename,
                "semi-lifted-twohands",
                "closed",
                "closed",
                "picking",
            ]

            if (
                behavior == "one-hand"
                or behavior == "diagonally"
                or behavior == "onehand-dropping"
                or behavior == "onehand-lifting"
            ):
                newData = [
                    obsFilename,
                    "semi-lifted-onehand",
                    "free",
                    "closed",
                    "picking",
                ]

            data.append(newData)
            episodeAcs.append(actionRescaled)
            episodeObs.append(obs)
            episodeInfo.append(info)

            timeStep += 1

        while True:
            print("PICKING UP", timeStep)
            if render:
                env.render(mode=render_mode)
            obsDataNew = obs.copy()
            objectPos = np.array(
                [
                    obsDataNew["observation"][7:10].copy(),
                    obsDataNew["observation"][10:13].copy(),
                    obsDataNew["observation"][13:16].copy(),
                    obsDataNew["observation"][16:19].copy(),
                ]
            )
            gripperPos = obsDataNew["observation"][:3].copy()
            gripperState = obsDataNew["observation"][3]

            object_rel_pos = objectPos - gripperPos

            gripperPos_2 = obsDataNew["observation"][19:22].copy()
            gripperState_2 = obsDataNew["observation"][22]

            object_oriented_goal = (
                obsDataNew["desired_goal"].copy()[3:6] - [0, 0.15, 0.3] - gripperPos
            )

            object_oriented_goal[1] += 0.1

            object_oriented_goal_2 = (
                obsDataNew["desired_goal"].copy()[0:3] - [0, 0.15, 0.3] - gripperPos_2
            )

            object_oriented_goal_2[1] += 0.1

            if behavior == "complex":
                if (
                    np.linalg.norm(object_oriented_goal) <= drop_threshold
                    and np.linalg.norm(object_oriented_goal_2) <= drop_threshold
                ) or timeStep >= max_episode_steps:
                    break

            action = [
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(0.6, 1.0),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(-0.00001, 0.00001),
                random.uniform(0.6, 1.0),
            ]

            if place_pos == 3:
                speed = 0.356
            else:
                speed = 0.856  # cap action to whatever speed you want
            speed = 0.556 + np.random.uniform(-0.3, 0.3)
            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]
                # action[i] = 0.

            for i in range(len(object_oriented_goal_2)):
                if behavior == "onehand-lowering":
                    action[4 + i] = 0.0
                else:
                    action[4 + i] = object_oriented_goal_2[i]

            actionRescaled = rescale_action(action, speed, noise_param)
            actionRescaled[3] = action[3]
            actionRescaled[7] = action[7]
            obs, reward, done, info = env.step(actionRescaled)
            obsDataNew = obs.copy()
            obsFilename = obsDataNew["filename"]

            newData = [
                obsFilename,
                "semi-lifted-twohands",
                "closed",
                "closed",
                "picking",
            ]

            if (
                behavior == "one-hand"
                or behavior == "diagonally"
                or behavior == "onehand-dropping"
            ):
                newData = [
                    obsFilename,
                    "semi-lifted-onehand",
                    "free",
                    "closed",
                    "picking",
                ]

            liftedFlag = all(i > 0.008 for i in (temp_obs[:, 2] - init_objectPos[:, 2]))

            if liftedFlag:
                episode_flagLifted = True

            if liftedFlag:
                if (
                    behavior == "onehand-lifting"
                    or behavior == "onehand-dropping"
                    or behavior == "onehand-lowering"
                ):
                    newData = [
                        obsFilename,
                        "lifted-onehand",
                        "free",
                        "closed",
                        "lifting_onehand",
                    ]
                else:
                    newData = [
                        obsFilename,
                        "lifted-twohands",
                        "closed",
                        "closed",
                        "lifting_two_hands",
                    ]

            data.append(newData)

            episodeAcs.append(actionRescaled)
            episodeObs.append(obs)
            episodeInfo.append(info)

            timeStep += 1

        while (timeStep) < max_episode_steps:
            if render:
                env.render(mode=render_mode)
            actionDull = [
                random.uniform(-0.00000001, 0.00000001),
                random.uniform(-0.00000001, 0.00000001),
                random.uniform(-0.00000001, 0.000000001),
                random.uniform(0.6, 1.0),
                random.uniform(-0.00000001, 0.00000001),
                random.uniform(-0.00000001, 0.00000001),
                random.uniform(-0.000001, 0.00000001),
                random.uniform(0.6, 1.0),
            ]
            actionRescaled = rescale_action(actionDull, 1.0, 0.0)
            obs, reward, done, info = env.step(actionRescaled)
            obsDataNew = obs.copy()
            obsFilename = obsDataNew["filename"]

            newData = [
                obsFilename,
                "semi-lifted-twohands",
                "closed",
                "closed",
                "picking",
            ]

            if behavior == "one-hand" or behavior == "diagonally":
                newData = [
                    obsFilename,
                    "semi-lifted-onehand",
                    "free",
                    "closed",
                    "picking",
                ]

            data.append(newData)

            episodeAcs.append(actionRescaled)
            episodeObs.append(obs)
            episodeInfo.append(info)
            timeStep += 1
    while True:
        if render:
            env.render(mode=render_mode)
        obsDataNew = obs.copy()
        objectPos = np.array(
            [
                obsDataNew["observation"][7:10].copy(),
                obsDataNew["observation"][10:13].copy(),
                obsDataNew["observation"][13:16].copy(),
                obsDataNew["observation"][16:19].copy(),
            ]
        )
        gripperPos = obsDataNew["observation"][:3].copy()
        gripperState = obsDataNew["observation"][3]

        object_rel_pos = objectPos - gripperPos

        gripperPos_2 = obsDataNew["observation"][19:22].copy()
        gripperState_2 = obsDataNew["observation"][22]

        object_oriented_goal_2 = obsDataNew["desired_goal"].copy()[0:3] - gripperPos_2

        if behavior == "diagonally":
            object_oriented_goal = obsDataNew["desired_goal"].copy() - gripperPos
        else:
            object_oriented_goal = obsDataNew["desired_goal"].copy()[3:6] - gripperPos

        object_oriented_goal[2] += 0.1

        object_oriented_goal[1] += 0.01

        object_oriented_goal_2[2] += 0.1

        object_oriented_goal_2[1] += 0.01

        if behavior == "one-hand" or behavior == "diagonally":
            if (
                np.linalg.norm(object_oriented_goal)
            ) <= reach_threshold or timeStep >= max_episode_steps:
                break
        elif (
            np.linalg.norm(object_oriented_goal) <= reach_threshold
            and np.linalg.norm(object_oriented_goal_2) <= reach_threshold
        ) or timeStep >= max_episode_steps:
            break

        action = [
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(0.6, 1.0),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(0.6, 1.0),
        ]
        if place_pos == 3:
            speed = 0.256
        else:
            speed = 0.556 + np.random.uniform(
                -0.3, 0.3
            )  # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]

        for i in range(len(object_oriented_goal_2)):
            if (
                behavior == "one-hand"
                or behavior == "diagonally"
                or behavior == "onehand-lifting"
            ):
                action[4 + 1] = 0
            else:
                action[4 + i] = object_oriented_goal_2[i]

        actionRescaled = rescale_action(action, speed, noise_param)
        actionRescaled[3] = action[3]
        actionRescaled[7] = action[7]
        obs, reward, done, info = env.step(actionRescaled)
        obsDataNew = obs.copy()
        obsFilename = obsDataNew["filename"]

        newData = [obsFilename, "semi-lifted-twohands", "closed", "closed", "taking"]

        if behavior == "one-hand" or behavior == "diagonally":
            newData = [obsFilename, "semi-lifted-onehand", "free", "closed", "taking"]

        liftedFlag = all(i > 0.008 for i in (temp_obs[:, 2] - init_objectPos[:, 2]))

        if liftedFlag:
            episode_flagLifted = True

        if liftedFlag:
            if behavior == "onehand-lifting" or behavior == "onehand-lowering":
                newData = [
                    obsFilename,
                    "lifted-onehand",
                    "free",
                    "closed",
                    "lifting_onehand",
                ]
            else:
                newData = [
                    obsFilename,
                    "lifted-twohands",
                    "closed",
                    "closed",
                    "lifting_two_hands",
                ]

        data.append(newData)

        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1
    while True:
        if render:
            env.render(mode=render_mode)
        obsDataNew = obs.copy()
        objectPos = np.array(
            [
                obsDataNew["observation"][7:10].copy(),
                obsDataNew["observation"][10:13].copy(),
                obsDataNew["observation"][13:16].copy(),
                obsDataNew["observation"][16:19].copy(),
            ]
        )
        gripperPos = obsDataNew["observation"][:3].copy()
        gripperState = obsDataNew["observation"][3]

        gripperPos_2 = obsDataNew["observation"][19:22].copy()
        gripperState_2 = obsDataNew["observation"][22]

        object_rel_pos = objectPos - gripperPos

        if behavior == "diagonally":
            object_oriented_goal = obsDataNew["desired_goal"].copy() - gripperPos
        else:
            object_oriented_goal = obsDataNew["desired_goal"].copy()[3:6] - gripperPos

        object_oriented_goal[2] += 0.02
        object_oriented_goal[1] += 0.02

        object_oriented_goal_2 = obsDataNew["desired_goal"].copy()[0:3] - gripperPos_2

        object_oriented_goal_2[2] += 0.02

        object_oriented_goal_2[1] += 0.02

        if behavior == "one-hand" or behavior == "diagonally":
            if (
                np.linalg.norm(object_oriented_goal)
            ) <= reach_threshold or timeStep >= max_episode_steps:
                break
        elif (
            np.linalg.norm(object_oriented_goal) <= reach_threshold
            and np.linalg.norm(object_oriented_goal_2) <= reach_threshold
        ) or timeStep >= max_episode_steps:
            break

        action = [
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(0.6, 1.0),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(0.6, 1.0),
        ]
        if place_pos == 3:
            speed = 0.256
        else:
            speed = 0.556 + np.random.uniform(
                -0.3, 0.3
            )  # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]

        for i in range(len(object_oriented_goal_2)):
            if behavior == "one-hand" or behavior == "diagonally":
                action[4 + 1] = 0
            else:
                action[4 + i] = object_oriented_goal_2[i]

        actionRescaled = rescale_action(action, speed, 0.4)
        obs, reward, done, info = env.step(actionRescaled)
        obsDataNew = obs.copy()
        obsFilename = obsDataNew["filename"]

        newData = [obsFilename, "semi-lifted-twohands", "closed", "closed", "taking"]

        if behavior == "one-hand" or behavior == "diagonally":
            newData = [obsFilename, "semi-lifted-onehand", "free", "closed", "taking"]

        liftedFlag = all(i > 0.008 for i in (temp_obs[:, 2] - init_objectPos[:, 2]))
        if liftedFlag:
            episode_flagLifted = True

        if liftedFlag:
            if behavior == "onehand-lifting" or behavior == "onehand-lowering":
                newData = [
                    obsFilename,
                    "lifted-onehand",
                    "free",
                    "closed",
                    "lifting_onehand",
                ]
            else:
                newData = [
                    obsFilename,
                    "lifted-twohands",
                    "closed",
                    "closed",
                    "lifting_two_hands",
                ]

        data.append(newData)

        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1

    while (timeStep) < max_episode_steps:
        obsDataNew = obs.copy()

        pdb.set_trace()

        objectPos = np.array(
            [
                obsDataNew["observation"][7:10].copy(),
                obsDataNew["observation"][10:13].copy(),
                obsDataNew["observation"][13:16].copy(),
                obsDataNew["observation"][16:19].copy(),
            ]
        )

        if render:
            env.render(mode=render_mode)
        actionDull = [
            random.uniform(-0.00000001, 0.00000001),
            random.uniform(-0.00000001, 0.00000001),
            random.uniform(0.0, 1.0),
            random.uniform(-0.00000001, 0.00000001),
            random.uniform(-0.00000001, 0.00000001),
            random.uniform(-0.00000001, 0.00000001),
            random.uniform(0.0, 1.0),
            random.uniform(-0.00000001, 0.00000001),
        ]
        actionRescaled = rescale_action(actionDull, 1.0, 0.0)
        obs, reward, done, info = env.step(actionRescaled)
        obsDataNew = obs.copy()
        obsFilename = obsDataNew["filename"]

        newData = [obsFilename, "folded", "free", "free", "waiting"]

        if behavior == "diagonally":
            newData = [obsFilename, "diagonally Folded", "free", "free", "waiting"]

        liftedFlag = all(i > 0.008 for i in (temp_obs[:, 2] - init_objectPos[:, 2]))
        if liftedFlag:
            episode_flagLifted = True

        if liftedFlag:
            if behavior == "onehand-lifting" or behavior == "onehand-lowering":
                newData = [
                    obsFilename,
                    "lifted-onehand",
                    "free",
                    "closed",
                    "lifting_onehand",
                ]
            else:
                newData = [
                    obsFilename,
                    "lifted-twohands",
                    "closed",
                    "closed",
                    "lifting_two_hands",
                ]

        data.append(newData)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)
        timeStep += 1

    with open(DIR + "data.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)

    return [episodeAcs, episodeObs, episodeInfo]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env",
        type=str,
        default="Gen3-v0",
        help="name of the environment. Options: Gen3-v0",
    )
    parser.add_argument(
        "--mode",
        choices=["noop", "random", "human", "demo"],
        default="random",
        help="mode of the agent",
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="maximum episode length"
    )
    parser.add_argument("--fps", type=float)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ignore_done", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--behavior",
        choices=[
            "diagonally",
            "sideways",
            "lifting",
            "dropping",
            "lowering",
            "one-hand",
            "onehand-lifting",
            "onehand-dropping",
            "onehand-lowering",
            "complex",
        ],
        default="sideways",
    )
    args = parser.parse_args()

    env = RandomizedEnvWrapper(envs.make(args.env), seed=1)
    action_space = env.action_space
    mode = args.mode
    render = args.render
    behavior = args.behavior
    print("Input render", render, args.render)
    fps = args.fps or env.metadata.get("video.frames_per_second") or 100
    if args.max_steps == 0:
        args.max_steps = env.spec.tags["wrapper_config.TimeLimit.max_episode_steps"]
        print("max_steps = ", args.max_steps)

    print("Press ESC to quit")
    reward = 0
    done = False

    if mode == "demo":
        actions = []
        observations = []
        infos = []
        numItr = 5
        fileName = (
            "data_mujoco"
            + "_"
            + "fold_sideways"
            + "_"
            + str(numItr)
            + "_T_100_"
            + "L_11_"
            + "all_randomized_explicit"
            ".npz"
        )

        actionDull = [
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
        ]
        obs = env.reset()
        if render:
            image_output = env.render(mode=render_mode)

        print("Starting a new trajectory")
        max_episode_steps = args.max_steps
        traj_success = 0
        while len(actions) < numItr:
            episodeAcs, episodeObs, episodeInfo = generate_demos(
                obs, render, max_episode_steps, behavior
            )
            actions.append(episodeAcs)
            observations.append(episodeObs)
            env.randomize(
                [
                    "random",
                    "random",
                    "random",
                    "random",
                    "random",
                    "random",
                    "random",
                    "random",
                    "random",
                ]
            )
            obs = env.reset()
            if render:
                image_output = env.render(mode=render_mode)  # default mode is human
            infos.append(episodeInfo)
            summ = 0
            print("ITERATION NUMBER ", len(actions))
