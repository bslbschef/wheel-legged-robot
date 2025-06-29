import pytest
from wheel_robot.pd_controller import PDController


def test_pd_controller_zero_error():
    controller = PDController(kp=1.0, kd=0.1)
    output = controller.compute(target=1.0, current=1.0)
    assert output == pytest.approx(0.0)


def test_pd_controller_basic_response():
    controller = PDController(kp=2.0, kd=0.5)
    out = controller.compute(target=2.0, current=1.0, target_velocity=0.0, current_velocity=-1.0)
    expected = 2.0 * (2.0 - 1.0) + 0.5 * (0.0 - (-1.0))
    assert out == pytest.approx(expected)
