import torch

def quat2mat_torch(quat, eps=0.0):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    norm_quat = quat.norm(p=2, dim=1, keepdim=True)
    # print('quat', quat) # Bx4
    # print('norm_quat: ', norm_quat)  # Bx1
    norm_quat = quat / (norm_quat + eps)
    # print('normed quat: ', norm_quat)
    qw, qx, qy, qz = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)

    s = 2.0  # * Nq = qw*qw + qx*qx + qy*qy + qz*qz
    X = qx * s
    Y = qy * s
    Z = qz * s
    wX = qw * X
    wY = qw * Y
    wZ = qw * Z
    xX = qx * X
    xY = qx * Y
    xZ = qx * Z
    yY = qy * Y
    yZ = qy * Z
    zZ = qz * Z
    rotMat = torch.stack(
        [1.0 - (yY + zZ), xY - wZ, xZ + wY, xY + wZ, 1.0 - (xX + zZ), yZ - wX, xZ - wY, yZ + wX, 1.0 - (xX + yY)], dim=1
    ).reshape(B, 3, 3)

    # rotMat = torch.stack([
    #     qw * qw + qx * qx - qy * qy - qz * qz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
    #     2 * (qx * qy + qw * qz), qw * qw - qx * qx + qy * qy - qz * qz, 2 * (qy * qz - qw * qx),
    #     2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz],
    #     dim=1).reshape(B, 3, 3)

    # w2, x2, y2, z2 = qw*qw, qx*qx, qy*qy, qz*qz
    # wx, wy, wz = qw*qx, qw*qy, qw*qz
    # xy, xz, yz = qx*qy, qx*qz, qy*qz

    # rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
    #                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
    #                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    # translation: Nx3
    # rot_allo: Nx3x3
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = torch.cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego