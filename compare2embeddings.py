import numpy as np
import matplotlib.pyplot as plt

def normalize_angle(angle):
    """
    Normalize an angle to the range [-π, π] using modulo arithmetic.

    This function ensures that input angles remain within a standard range
    by wrapping them using modulo operations.

    Args:
        angle (torch.Tensor): Angle in radians.

    Returns:
        torch.Tensor: Normalized angle in the range [-π, π].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def normalize_angle_smooth(angle):
    """
    Smoothly normalize an angle to the range [-π, π] using trigonometric functions.

    This approach ensures differentiability across all values, making it
    suitable for gradient-based optimization tasks.

    Args:
        angle (torch.Tensor): Angle in radians.

    Returns:
        torch.Tensor: Smoothly normalized angle in the range [-π, π].
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

def angular_difference(angle1, angle2, smooth: bool = True):
    """
    Compute the shortest angular difference between two angles in radians.

    The function calculates the absolute shortest distance between two angles
    while accounting for periodicity. Optionally, it can use a differentiable 
    normalization method.

    Args:
        angle1 (torch.Tensor): First angle in radians.
        angle2 (torch.Tensor): Second angle in radians.
        smooth (bool, optional): If True, uses a differentiable approach. Defaults to True.

    Returns:
        torch.Tensor: The absolute shortest distance between the two angles in radians.
    """
    diff = normalize_angle_smooth(angle2 - angle1) if smooth else normalize_angle(angle2 - angle1)
    return np.abs(diff)


if __name__ == "__main__":
    old_emb = np.load("Embeddings/pRotatE_1000_Entity_Embeddings_FB15k_reduced_n10_deg1.npy")
    new_emb = np.load("Embeddings/pRotatE_1000_Entity_Embeddings_FB15k_reduced_n10_deg1_merged.npy")
    # this one below is trained separately, maybe comparison is not needed
    full_emb = np.load("Embeddings/pRotatE_1000_Entity_Embeddings_FB15k.npy")
    
    print(f"OLD Embeddings shape:{old_emb.shape}")
    print(f"NEW Embeddings shape:{new_emb.shape}")

    # * Step 1
    # Compare if old_emb == new_emb[:new_entities]
    if np.all(old_emb == new_emb[:-10, :]):
        print("Between OLD and NEW Embeddings: All elements are equal.")
    else:
        print("Between OLD and NEW Embeddings: Not all elements are equal.")

    # * Step 2
    # Compare if new_emb[-10:] somewhat == full_emb[-10:]
    # for this plot the difference between 2
    # comprae difference between entity n, and m
    # for both new and full embeddings
    # plot them 2 figures

    ind1 = 0
    ind2 = 500

    embedding_range = 0.026000000536441803
    e1_new = new_emb[ind1]
    e2_new = new_emb[ind2]
    e1_full = full_emb[ind1]
    e2_full = full_emb[ind2]

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6)) # 1 row, 2 columns
    # # Plot on the first subplot (ax1)
    # ax1.plot(e1_new, label='NEW E1', color='blue')
    # ax1.plot(e2_new, label='NEW E2', color='red')
    # ax1.set_title('New Embeddings')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.legend()
    # ax1.grid(True)

    # ax2.plot(e1_full, label='FULL E1', color='blue')
    # ax2.plot(e2_full, label='FULL E2', color='red')
    # ax2.set_title('FULL Embeddings')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.legend()
    # ax2.grid(True)

    # plt.tight_layout()
    # plt.savefig("images/compare_new2old.png")
    # plt.close()

    # * Step 3
    ind1 = 14949
    difference_total = []
    difference_avg = []
    for ind2 in range(full_emb.shape[0]):
        embedding_range = 0.026000000536441803
        e1_new = new_emb[ind1]
        e2_new = new_emb[ind2]
        e1_full = full_emb[ind1]
        e2_full = full_emb[ind2]

        e1_new = e1_new/(embedding_range/np.pi)
        e2_new = e1_new/(embedding_range/np.pi)
        e1_full = e1_full/(embedding_range/np.pi)
        e2_full = e2_full/(embedding_range/np.pi)
        diff1 = angular_difference(e1_new, e2_new, smooth=True)
        diff2 = angular_difference(e1_full, e2_full, smooth=True)
        rotational_total1 = (180/np.pi)*(diff1.sum().item())
        rotational_total2 = (180/np.pi)*(diff2.sum().item())
        rotational_avg1 = (180/np.pi)*(diff1.mean().item())
        rotational_avg2 = (180/np.pi)*(diff2.mean().item())

        # print(f"Rotation Total Diff between New e1-e2 is {rotational_total1}")
        # print(f"Rotation Total Diff between Full e1-e2 is {rotational_total2}")
        difference_total.append(np.abs(rotational_total1-rotational_total2))

        # print(f"Rotation Avg Diff between New e1-e2 is {rotational_avg1}")
        # print(f"Rotation Avg Diff between Full e1-e2 is {rotational_avg2}")
        difference_avg.append(np.abs(rotational_avg1-rotational_avg2))

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot histogram of difference_avg on the first subplot (ax1)
    ax1.hist(difference_avg, bins=360, color='skyblue', edgecolor='black')
    ax1.set_title('Histogram of Difference (Average)')
    ax1.set_xlabel('Difference (Average)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.6)  # Add a grid

    # Plot histogram of difference_total on the second subplot (ax2)
    ax2.hist(difference_total, bins=100, color='salmon', edgecolor='black')
    ax2.set_title('Histogram of Difference (Total)')
    ax2.set_xlabel('Difference (Total)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.6)  # Add a grid

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()

    # Show the plot
    plt.savefig("images/histogram_rotation_diff.png")
    plt.close()
    
