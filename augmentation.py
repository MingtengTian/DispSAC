import numpy as np
import torch

def _masked_std_per_row(x, mask):
	"""Unbiased std along dim=1 using only mask==True entries. x, mask: [B, N] -> [B]."""
	m = mask.float()
	count = m.sum(dim=1)
	mean = (x * m).sum(dim=1) / count.clamp(min=1e-12)
	centered = (x - mean.unsqueeze(1)) * m
	sum_sq = (centered ** 2).sum(dim=1)
	denom = (count - 1).clamp(min=1)
	var = sum_sq / denom
	std = torch.sqrt(var.clamp(min=0.0))
	return torch.where(count > 1, std, torch.zeros_like(std))

def add_gaussian_noise_channel(input_data, noise_level):
	"""
	Apply Gaussian noise to phase velocity and group velocity channels.

	Parameters:
	-----------
	input_data : torch.Tensor
		A 2D tensor where:
		- Row 0: Periods
		- Row 1: Phase velocities
		- Row 2: Group velocities

	noise_level : float, optional
		The noise level as a fraction of the standard deviation of the channel data (default is 0.05).

	Returns:
	--------
	torch.Tensor
		The input data with added Gaussian noise to phase and group velocity channels.
	"""
	# Mask invalid data points in phase and group velocities
	valid_phase_mask = input_data[..., 1] > 0
	valid_group_mask = input_data[..., 2] > 0

	# Per-row std over valid columns: shape [batch] (e.g. [96]); no torch.nanstd (old PyTorch)
	phase_vel_std = _masked_std_per_row(input_data[..., 1], valid_phase_mask)
	group_vel_std = _masked_std_per_row(input_data[..., 2], valid_group_mask)

	# Broadcast to [batch, n_periods] for elementwise noise (same std for all periods in a row)
	phase_vel_noise_std = noise_level * phase_vel_std.unsqueeze(1)
	group_vel_noise_std = noise_level * group_vel_std.unsqueeze(1)

	phase_delta = torch.randn_like(input_data[..., 1]) * phase_vel_noise_std
	group_delta = torch.randn_like(input_data[..., 2]) * group_vel_noise_std
	input_data[..., 1] = torch.where(
		valid_phase_mask,
		input_data[..., 1] + phase_delta,
		input_data[..., 1],
	)
	input_data[..., 2] = torch.where(
		valid_group_mask,
		input_data[..., 2] + group_delta,
		input_data[..., 2],
	)

	return input_data, phase_vel_noise_std, group_vel_noise_std


def edge_masking(input_data, max_ratio, masking_value=0):
	"""
	Independently mask Rayleigh and Love phase-velocity rows.
	Each row may be masked at the beginning, end, or both,
	with independent mask lengths. If 'both', the mask length
	is randomly split between front and end.
	"""
	n = input_data.shape[1]
	masked = input_data.clone().float()

	def mask_row(row_idx):
		mask_len = np.random.randint(1, int(min(max_ratio * n, n)) + 1)
		choice = np.random.choice(["begin", "end", "both"], p=[0.3, 0.2, 0.5])

		if choice == "begin":
			masked[:, :mask_len, row_idx] = masking_value

		elif choice == "end":
			masked[:, -mask_len:, row_idx] = masking_value

		else:  # both
			front_len = np.random.randint(0, mask_len + 1)
			end_len = mask_len - front_len

			if front_len > 0:
				masked[:, :front_len, row_idx] = masking_value
			if end_len > 0:
				masked[:, -end_len:, row_idx] = masking_value

	# Row 1: Rayleigh
	mask_row(1)

	# Row 2: Love
	mask_row(2)

	return masked


def augmentation(input_data, noise_level, max_cut_ratio):#, max_mask_ratio, max_cut_ratio, max_remove_ratio):
    if noise_level > 0:
        print(f"\n[Augmentation] Add {noise_level} dispersion noise at most")
        input_data, *_ = add_gaussian_noise_channel(input_data.clone(), noise_level)
    # if max_mask_ratio > 0:
    #     print(f"mask {max_mask_ratio} dispersion at most")
    #     input_data = random_masking(input_data.clone(), max_mask_ratio)
    if max_cut_ratio > 0:
        print(f"cut {max_cut_ratio} dispersion at most")
        input_data = edge_masking(input_data.clone(), max_cut_ratio)
    # if max_remove_ratio > 0:
    #     print(f"remove {max_remove_ratio} dispersion at most")
    #     input_data = remove_entire_sequence(input_data.clone(), max_remove_ratio)
    return input_data
