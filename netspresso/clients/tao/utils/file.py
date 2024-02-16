import os
import tarfile


def split_tar_file(input_tar_path, output_dir, max_split_size=200 * 1024 * 1024):
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(input_tar_path, "r") as original_tar:
        members = original_tar.getmembers()
        current_split_size = 0
        current_split_number = 0
        current_split_name = os.path.join(output_dir, f"smaller_file_{current_split_number}.tar")

        with tarfile.open(current_split_name, "w") as split_tar:
            for member in members:
                if current_split_size + member.size <= max_split_size:
                    split_tar.addfile(member, original_tar.extractfile(member))
                    current_split_size += member.size
                else:
                    split_tar.close()
                    current_split_number += 1
                    current_split_name = os.path.join(output_dir, f"smaller_file_{current_split_number}.tar")
                    current_split_size = 0
                    split_tar = tarfile.open(current_split_name, "w")  # Open a new split tar archive
                    split_tar.addfile(member, original_tar.extractfile(member))
                    current_split_size += member.size
