from lingbot_map.utils.load_fn import load_and_preprocess_images


def crop_max_height(images, max_height=None, patch_size=14):
    if max_height is None:
        return images
    if max_height < patch_size or max_height % patch_size:
        raise ValueError('max_height must be a positive patch-size multiple')
    height = images.shape[-2]
    top = max((height - max_height) // 2, 0)
    return images[..., top:top + min(height, max_height), :].contiguous()


class LazyImageSequence:
    """CPU, batch-one image slices for the original streaming entry point."""

    ndim = 5

    def __init__(self, paths, image_size=224, patch_size=14, max_height=None):
        self.paths = tuple(str(path) for path in paths)
        if not self.paths:
            raise ValueError('Lazy streaming requires at least one image')
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_height = max_height
        sample = self._load(self.paths[:1])
        self.shape = (1, len(self.paths), *sample.shape[1:])

    def _load(self, paths):
        images = load_and_preprocess_images(list(paths), image_size=self.image_size, patch_size=self.patch_size)
        return crop_max_height(images, self.max_height, self.patch_size)

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 2 or index[0] != slice(None):
            raise IndexError('Lazy images support only images[:, start:stop]')
        sequence_slice = index[1]
        if not isinstance(sequence_slice, slice) or sequence_slice.step not in (None, 1):
            raise IndexError('Lazy images require contiguous frame slices')
        paths = self.paths[sequence_slice]
        if not paths:
            raise ValueError('Empty lazy image slice')
        images = self._load(paths)
        if images.shape[1:] != self.shape[2:]:
            raise ValueError('Lazy streaming requires a fixed preprocessed image shape; use eager loading for mixed sizes')
        return images.unsqueeze(0)
