from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.coco_loader import COCOLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'pascal': pascalVOCLoader,
        'sbd': pascalVOCLoader,
        'coco': COCOLoader,
    }[name]
