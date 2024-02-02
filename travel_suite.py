import os
import torch
import numpy as np
from scipy.stats import norm
from PIL import Image

class LatentTravel:
    """Travel between two latent vectors"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "A": ("LATENT",),
                "B": ("LATENT",),
                "steps": ("INT", {"default": 5, "min": 3, "max": 10000, "step": 1}),
                "factor": ("FLOAT", {"default": 0.5}),
                "blend_mode": ( ["lerp", "slerp", "add", "multiply", "divide", "subtract", "overlay", "hard_light",
                           "soft_light", "screen", "linear_dodge", "difference", "exclusion", "random"],),
                "travel_mode": ( ['linear', 'hinge', 'circle', 'norm', 'quadratic', 'cubic', 'quartic', 'geometric'],),
                "reflect_travel": ("BOOLEAN", {"default": True}),
                "vae": ("VAE",),
                "output_images": ("BOOLEAN", {"default": False}),
                "filepath": ("STRING", {"default": "output/travel"}),
                "prefix": ("STRING", {"default": "travel"}),
                "write_images": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("LATENTS", "IMAGES", "FILEPATHS")
    FUNCTION = "latent_travel"
    CATEGORY = "travel/latent"

    def latent_travel(self, A, B, steps, factor, vae, blend_mode, travel_mode, reflect_travel,
                      filepath, prefix, write_images, output_images):

        out_paths = []
        out_images = []
        out_latents = []

        # Get cutpoints based on travel mode
        cutpoints = self.get_travel_cutpoints(0, 1, steps, travel_mode, factor, reflect_travel)

        # Blend latents using travel cutpoints and blend mode
        out_latents = [self.blend_latents(B['samples'], A['samples'], blend_mode, t) for t in cutpoints]
        out_latents = torch.cat(out_latents, 0)

        if output_images:
            out_images = vae.decode(out_latents)

        if write_images:
            print(f'Writing latent travel images to: {filepath}')
            os.makedirs(filepath, exist_ok=True)
            for index, img in enumerate(out_images):
                out_paths.append(self.save_image(img, filepath, prefix, index))

        return ({'samples': out_latents}, out_images, out_paths)

    def powspace(self, start, stop, power, steps):
        start = np.power(start, 1 / float(power))
        stop = np.power(stop, 1 / float(power))
        return np.power(np.linspace(start, stop, num=steps), power)

    def get_travel_cutpoints(self, start, stop, steps, travel_mode, factor, reflect_travel):

        if travel_mode == 'linear':
            return np.linspace(0, 1, steps)
        elif travel_mode == 'hinge':
            cutpoints = hinge_points(start, stop, steps, factor)
        elif travel_mode == 'circle':
            cutpoints = circle_points(steps)
        elif travel_mode == 'norm':
            cutpoints = normspace(start, stop, steps, factor)
        elif travel_mode == 'quadratic':
            cutpoints = quadraticspace(start, stop, steps)
            cutpoints = reflect_values(cutpoints) if reflect_travel else cutpoints
        elif travel_mode == 'cubic':
            cutpoints = cubicspace(start, stop, steps)
            cutpoints = reflect_values(cutpoints) if reflect_travel else cutpoints
        elif travel_mode == 'quartic':
            cutpoints = quarticspace(start, stop, steps)
            cutpoints = reflect_values(cutpoints) if reflect_travel else cutpoints
        elif travel_mode == 'geometric':
            cutpoints = geomspace(start, stop, steps)
            cutpoints = reflect_values(cutpoints) if reflect_travel else cutpoints
        else:
            raise ValueError(f"Unsupported travel mode {travel_mode}. "
                             f"Please choose from 'linear', 'hinge', 'circle', 'norm', 'quadratic', "
                             f"'quartic', 'geometric'")
        return cutpoints

    def blend_latents(self, A, B, mode='lerp', factor=0.5):

        factor1 = factor
        factor2 = 1 - factor

        if mode == 'lerp':
            out = lerp(A, B, factor1)
        elif mode == 'slerp':
            out = slerp(A, B, factor1)
        elif mode == 'add':
            out = (A * factor1) + (B * factor2)
        elif mode == 'multiply':
            out = (A * factor1) * (B * factor2)
        elif mode == 'divide':
            out = (A * factor1) / (B * factor2)
        elif mode == 'subtract':
            out = (A * factor1) - (B * factor2)
        elif mode == 'overlay':
            out = overlay_blend(A, B, factor1)
        elif mode == 'screen':
            out = screen_blend(A, B, factor1)
        elif mode == 'difference':
            out = difference_blend(A, B, factor1)
        elif mode == 'exclusion':
            out = exclusion_blend(A, B, factor1)
        elif mode == 'hard_light':
            out = hard_light_blend(A, B, factor1)
        elif mode == 'linear_dodge':
            out = linear_dodge_blend(A, B, factor1)
        elif mode == 'soft_light':
            out = soft_light_blend(A, B, factor1)
        elif mode == 'random':
            out = random_noise(A, B, factor1)
        else:
            raise ValueError(
                f"Unsupported blending mode {mode}. "
                f"Please choose from 'add', 'multiply', 'divide', 'subtract', 'overlay', 'screen', "
                f"'difference', 'exclusion', 'hard_light', 'linear_dodge', 'soft_light', 'custom_noise'.")

        return out

    def save_image(self, tensor, filepath, prefix, index):

        image = tensor2pil(tensor)

        filename = f'{prefix}_{index:05}.png'
        output_path = os.path.join(filepath, filename)
        image.save(output_path)
        return output_path


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


################
# TRAVEL MODES #
################

def powspace(start, stop, power, steps):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=steps), power)


def geomspace(start, stop, steps):
    X = np.geomspace(start + 1e-10, stop, steps)
    X[0] = 0
    return X


def quadraticspace(start, stop, steps):
    return powspace(start, stop, 2, steps)


def cubicspace(start, stop, steps):
    return powspace(start, stop, 3, steps)


def quarticspace(start, stop, steps):
    return powspace(start, stop, 4, steps)


def reflect_values(X):
    Y = 1 - np.flip(X)
    total_steps = len(X)
    reflect_step = total_steps // 2

    X_a = X[:reflect_step]
    X_b = np.flip(1 - X_a)

    if total_steps % 2 == 0:
        out = np.concatenate([X_a, X_b])
    else:
        mid = total_steps // 2
        mean_val = np.mean([X[mid], Y[mid]]).reshape(1)
        out = np.concatenate([X_a, mean_val, X_b])
    return np.round(out, 5)


def circle_points(steps):
    # Angle in radians from start to stop
    start_angle = np.radians(180)
    stop_angle = np.radians(0)

    # Linspace to get 'steps' number of points between start_angle and stop_angle
    theta = np.linspace(start_angle, stop_angle, steps)

    # x and y coordinates for the points on the circle
    x = (np.cos(theta) + 1) / 2
    # y = np.sin(theta)

    return x


def hinge_points(start, stop, steps, hinge):
    if steps % 2 == 0:
        A = np.linspace(start, hinge, num=steps // 2)
        B = np.linspace(hinge, stop, num=(steps // 2 + 1))
        out = np.concatenate([A, B[1:]])  # remove duplicated end point
    else:
        A = np.linspace(start, hinge, num=steps // 2)
        B = np.linspace(hinge, stop, num=(steps // 2) + 2)
        out = np.concatenate([A, B[1:]])  # remove duplicated end point

    return out


def normspace(start, stop, steps, factor):
    X = np.linspace(start, stop, int(steps - 2))
    Y = norm.cdf(X, loc=0.5, scale=factor)
    # Insert 0 and 1 strengths so that starting and end images are unchanged
    Y = np.insert(Y, 0, 0)
    Y = np.append(Y, 1)

    return Y

###################
# LATENT BLENDING #
###################

def lerp(B, A, factor):
    out = torch.lerp(A, B, factor)
    return out

def slerp(B, A, factor):
    # from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
    dims = A.shape
    A = A.reshape(dims[0], -1)  # flatten to batches
    B = B.reshape(dims[0], -1)
    low_norm = A / torch.norm(A, dim=1, keepdim=True)
    high_norm = B / torch.norm(B, dim=1, keepdim=True)
    low_norm[low_norm != low_norm] = 0.0  # in case we divide by zero
    high_norm[high_norm != high_norm] = 0.0
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - factor) * omega) / so).unsqueeze(1) * A + (
            torch.sin(factor * omega) / so).unsqueeze(1) * B
    return res.reshape(dims)

def overlay_blend(A, B, factor):
    low = 2 * A * B
    high = 1 - 2 * (1 - A) * (1 - B)
    blended_latent = (A * factor) * low + (B * factor) * high
    return blended_latent

def screen_blend(A, B, factor):
    inverted_A = 1 - A
    inverted_B = 1 - B
    blended_latent = 1 - (inverted_A * inverted_B * (1 - factor))
    return blended_latent

def difference_blend(A, B, factor):
    blended_latent = abs(A - B) * factor
    return blended_latent

def exclusion_blend(A, B, factor):
    blended_latent = (A + B - 2 * A * B) * factor
    return blended_latent

def hard_light_blend(A, B, factor):
    blended_latent = torch.where(B < 0.5, 2 * A * B,
                                 1 - 2 * (1 - A) * (1 - B)) * factor
    return blended_latent

def linear_dodge_blend(A, B, factor):
    blended_latent = torch.clamp(A + B, 0, 1) * factor
    return blended_latent

def soft_light_blend(A, B, factor):
    low = 2 * A * B + A ** 2 - 2 * A * B * A
    high = 2 * A * (1 - B) + torch.sqrt(A) * (2 * B - 1)
    blended_latent = (A * factor) * low + (B * factor) * high
    return blended_latent

def random_noise(A, B, factor):
    noise1 = torch.randn_like(A)
    noise2 = torch.randn_like(B)
    noise1 = (noise1 - noise1.min()) / (noise1.max() - noise1.min())
    noise2 = (noise2 - noise2.min()) / (noise2.max() - noise2.min())
    blended_noise = (A * factor) * noise1 + (B * factor) * noise2
    blended_noise = torch.clamp(blended_noise, 0, 1)
    return blended_noise


NODE_CLASS_MAPPINGS = {
    "LatentTravel": LatentTravel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentTravel": "Latent Travel",
}

# class BilateralFilter:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"images": ("IMAGE", ),
#                              "kernel_size": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
#                              "sigma_color": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.01}),
#                              "sigma_space": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.01}),
#                              }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "bilateral_filter"
#
#     CATEGORY = "ImageProcessing"
#
#     def bilateral_filter(self, images, kernel_size, sigma_color, sigma_space):
#         images = images.movedim(-1, 1).cpu()
#         images_transformed = bilateral_blur(images, (kernel_size, kernel_size), sigma_color, (sigma_space, sigma_space), color_distance_type="l2")
#         images_transformed = images_transformed.movedim(1, -1)
#
#         return (images_transformed,)
#
#
# class UnsharpMask:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"images": ("IMAGE", ),
#                              "kernel_size": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
#                              "sigma": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.01}),
#                              }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "sharpen"
#
#     CATEGORY = "ImageProcessing"
#
#     def sharpen(self, images, kernel_size, sigma):
#         images = images.movedim(-1, 1).cpu()
#         images_transformed = unsharp_mask(images, (kernel_size, kernel_size), (sigma, sigma))
#         images_transformed = images_transformed.movedim(1, -1)
#
#         return (images_transformed,)
#
#
# class Hue:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"images": ("IMAGE", ),
#                              "factor": ("FLOAT", {"default": 0.0, "min": -3.141516, "max": 3.141516, "step": 0.001}),
#                              }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "hue"
#
#     CATEGORY = "ImageProcessing"
#
#     def hue(self, images, factor):
#         images = images.movedim(-1, 1).cpu()
#         images_transformed = adjust_hue(images, factor)
#         images_transformed = images_transformed.movedim(1, -1)
#
#         return (images_transformed,)
#
#
# class Saturation:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"images": ("IMAGE", ),
#                              "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
#                              }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "saturation"
#
#     CATEGORY = "ImageProcessing"
#
#     def saturation(self, images, factor):
#         images = images.movedim(-1, 1).cpu()
#         images_transformed = adjust_saturation(images, factor)
#         images_transformed = images_transformed.movedim(1, -1)
#
#         return (images_transformed,)
#
#
# class Brightness:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"images": ("IMAGE", ),
#                              "factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
#                              }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "brightness"
#
#     CATEGORY = "ImageProcessing"
#
#     def brightness(self, images, factor):
#         images = images.movedim(-1, 1).cpu()
#         images_transformed = adjust_brightness(images, factor)
#         images_transformed = images_transformed.movedim(1, -1)
#
#         return (images_transformed,)
#
#
# class Gamma:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"images": ("IMAGE", ),
#                              "gamma_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
#                              }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "gamma"
#
#     CATEGORY = "ImageProcessing"
#
#     def gamma(self, images, gamma_value):
#         images = images.movedim(-1, 1).cpu()
#         images_transformed = adjust_gamma(images, gamma_value)
#         images_transformed = images_transformed.movedim(1, -1)
#
#         return (images_transformed,)
#
#
# class SigmoidCorrection:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"images": ("IMAGE", ),
#                              "cutoff": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
#                              "gain": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.01}),
#                              }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "sigmoid"
#
#     CATEGORY = "ImageProcessing"
#
#     def sigmoid(self, images, cutoff, gain):
#         images = images.movedim(-1, 1).cpu()
#         images_transformed = adjust_sigmoid(images, cutoff, gain)
#         images_transformed = images_transformed.movedim(1, -1)
#
#         return (images_transformed,)
