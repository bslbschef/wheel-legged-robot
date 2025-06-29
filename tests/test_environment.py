from wheel_robot.environment import WheelLeggedEnv


def test_environment_initialization():
    env = WheelLeggedEnv()
    assert env.initialized
    assert env.reset() == 0
