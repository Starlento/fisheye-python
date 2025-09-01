import cv2
import numpy as np


def _default_fisheye_params(img_shape):
    h, w = img_shape[:2]
    return dict(f=min(w, h) * 0.5, cx=w * 0.5, cy=h * 0.5)


def fisheye_convert(img, k1, k2, fisheye=None):
    """
    Convert a fisheye image from factor k1 to k2 using r = f * theta^k.

    Parameters
    ----------
    img : np.ndarray
        Input fisheye image.
    k1 : float
        Source fisheye factor.
    k2 : float
        Target fisheye factor.
    fisheye : dict, optional
        Dict with keys {f, cx, cy} for source/target fisheye intrinsics.
        Defaults: f = 0.5 * min(W,H), cx=W/2, cy=H/2.

    Returns
    -------
    np.ndarray
        Converted image (same size as input).
    """
    h, w = img.shape[:2]
    if fisheye is None:
        fisheye = _default_fisheye_params(img.shape)
    f = float(fisheye.get("f"))
    cx = float(fisheye.get("cx"))
    cy = float(fisheye.get("cy"))

    # Destination grid (target k2)
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    dx = xv - cx
    dy = yv - cy
    r2 = np.sqrt(dx * dx + dy * dy)

    # From target radius (k2) -> angle -> source radius (k1)
    eps = 1e-8
    theta = (r2 / (f + eps)) ** (1.0 / max(k2, eps))
    r1 = f * (theta**k1)

    scale = r1 / (r2 + eps)
    map_x = scale * dx + cx
    map_y = scale * dy + cy

    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def fisheye_to_pinhole(img, k, pinhole, fisheye=None):
    """
    Reproject a fisheye image (factor k) into a pinhole camera image.

    Model assumptions:
        Fisheye: r_fish = f_fish * theta^k
        Pinhole: normalized coords (x = (u-cx)/fx, y = (v-cy)/fy), ray ~ (x,y,1)
                 theta = arctan( sqrt(x^2 + y^2) )

    Parameters
    ----------
    img : np.ndarray
        Input fisheye image.
    k : float
        Fisheye factor of the input image.
    pinhole : dict
        Target pinhole intrinsics and size:
            - fx, fy : focal lengths (pixels)
            - cx, cy : principal point (pixels)
            - width, height : output image size (pixels)
        Example: dict(fx=800, fy=800, cx=640, cy=360, width=1280, height=720)
    fisheye : dict, optional
        Source fisheye intrinsics {f, cx, cy}.
        Defaults to center = image center and f = 0.5 * min(W,H).

    Returns
    -------
    np.ndarray
        Pinhole-projected image of size (height, width, 3).
    """
    if fisheye is None:
        fisheye = _default_fisheye_params(img.shape)
    f_fish_x = float(fisheye.get("fx"))
    f_fish_y = float(fisheye.get("fy"))
    cx_f = float(fisheye.get("cx"))
    cy_f = float(fisheye.get("cy"))

    fx = float(pinhole["fx"])
    fy = float(pinhole["fy"])
    cx_p = float(pinhole["cx"])
    cy_p = float(pinhole["cy"])
    W = int(pinhole["width"])
    H = int(pinhole["height"])

    # Build destination (pinhole) grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Normalized pinhole coords -> ray angle θ
    x = (uu - cx_p) / fx
    y = (vv - cy_p) / fy
    rho = np.sqrt(x * x + y * y)  # = tan(theta)
    theta = np.arctan(rho)  # angle from optical axis

    # Map θ to fisheye radius using k
    r_fish_x = f_fish_x * (np.maximum(theta, 0.0) ** k)  # r = f * θ^k
    r_fish_y = f_fish_y * (np.maximum(theta, 0.0) ** k)  # r = f * θ^k

    # Direction on fisheye image plane (assume same optical axis; no rotation)
    # For rho=0, set unit direction arbitrary (no effect since r_fish=0)
    eps = 1e-8
    dir_x = np.where(rho > eps, x / (rho + eps), 0.0)
    dir_y = np.where(rho > eps, y / (rho + eps), 0.0)

    map_x = dir_x * r_fish_x + cx_f
    map_y = dir_y * r_fish_y + cy_f

    # Sample from fisheye source into pinhole target
    out = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return out


def pinhole_to_fisheye(img, k, pinhole, fisheye=None):
    """
    Project a pinhole image into a fisheye image (factor k).

    Model assumptions:
        Pinhole: normalized coords (x = (u-cx)/fx, y = (v-cy)/fy), ray ~ (x,y,1)
                 theta = arctan(sqrt(x^2 + y^2))
        Fisheye: r_fish = f_fish * theta^k

    Parameters
    ----------
    img : np.ndarray
        Input pinhole image.
    k : float
        Target fisheye factor.
    pinhole : dict
        Source pinhole intrinsics and size:
            - fx, fy : focal lengths (pixels)
            - cx, cy : principal point (pixels)
            - width, height : input image size (pixels)
    fisheye : dict, optional
        Target fisheye intrinsics {f, cx, cy, width, height}.
        Defaults to square fisheye (f = 0.5*min(W,H)) with center at (W/2,H/2).

    Returns
    -------
    np.ndarray
        Fisheye-projected image of size (height,width,3).
    """
    Hs, Ws = img.shape[:2]

    fx = float(pinhole["fx"])
    fy = float(pinhole["fy"])
    cx_p = float(pinhole["cx"])
    cy_p = float(pinhole["cy"])

    if fisheye is None:
        W = Ws
        H = Hs
        fisheye = dict(f=0.5 * min(W, H), cx=W / 2, cy=H / 2, width=W, height=H)
    f_fish = float(fisheye["f"])
    cx_f = float(fisheye["cx"])
    cy_f = float(fisheye["cy"])
    W = int(fisheye["width"])
    H = int(fisheye["height"])

    # Build destination (fisheye) grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    dx = uu - cx_f
    dy = vv - cy_f
    r = np.sqrt(dx * dx + dy * dy)

    # fisheye radius -> theta
    eps = 1e-8
    theta = (r / (f_fish + eps)) ** (1.0 / max(k, eps))

    # Convert theta to pinhole normalized coords
    tan_theta = np.tan(theta)
    dir_x = np.where(r > eps, dx / (r + eps), 0.0)
    dir_y = np.where(r > eps, dy / (r + eps), 0.0)

    x = dir_x * tan_theta
    y = dir_y * tan_theta

    # Back-project to pinhole pixel coords
    map_x = fx * x + cx_p
    map_y = fy * y + cy_p

    out = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return out
