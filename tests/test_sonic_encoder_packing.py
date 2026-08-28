import numpy as np

from unilab.envs.motion_tracking.g1.tracking_sonic import _pack_sonic_encoder_command


def test_sonic_encoder_command_preserves_legacy_temporal_reshape() -> None:
    joint_pos = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    joint_vel = (100 + np.arange(12, dtype=np.float32)).reshape(1, 3, 4)

    packed = _pack_sonic_encoder_command(joint_pos, joint_vel)

    expected = np.concatenate([joint_pos.reshape(1, -1), joint_vel.reshape(1, -1)], axis=1)
    np.testing.assert_array_equal(packed, expected.reshape(1, 3, 8))
    assert packed[0, 0].tolist() == list(range(8))
    assert packed[0, 1].tolist() == [8.0, 9.0, 10.0, 11.0, 100.0, 101.0, 102.0, 103.0]