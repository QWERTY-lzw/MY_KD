# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ImageNet100(CustomDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    The dataset supports two kinds of annotation format. More details can be
    found in :class:`CustomDataset`.

    Args:
        data_prefix (str): The path of data directory.
        pipeline (Sequence[dict]): A list of dict, where each element
            represents a operation defined in :mod:`mmcls.datasets.pipelines`.
            Defaults to an empty tuple.
        classes (str | Sequence[str], optional): Specify names of classes.

            - If is string, it should be a file path, and the every line of
              the file is a name of a class.
            - If is a sequence of string, every item is a name of class.
            - If is None, use the default ImageNet-1k classes names.

            Defaults to None.
        ann_file (str, optional): The annotation file. If is string, read
            samples paths from the ann_file. If is None, find samples in
            ``data_prefix``. Defaults to None.
        extensions (Sequence[str]): A sequence of allowed extensions. Defaults
            to ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif').
        test_mode (bool): In train mode or test mode. It's only a mark and
            won't be used in this class. Defaults to False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            If None, automatically inference from the specified path.
            Defaults to None.
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = [
        'tench, Tinca tinca',
        'goldfish, Carassius auratus',
        'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',  # noqa: E501
        'tiger shark, Galeocerdo cuvieri',
        'hammerhead, hammerhead shark',
        'electric ray, crampfish, numbfish, torpedo',
        'stingray',
        'cock',
        'hen',
        'ostrich, Struthio camelus',
        'brambling, Fringilla montifringilla',
        'goldfinch, Carduelis carduelis',
        'house finch, linnet, Carpodacus mexicanus',
        'junco, snowbird',
        'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
        'robin, American robin, Turdus migratorius',
        'bulbul',
        'jay',
        'magpie',
        'chickadee',
        'water ouzel, dipper',
        'kite',
        'bald eagle, American eagle, Haliaeetus leucocephalus',
        'vulture',
        'great grey owl, great gray owl, Strix nebulosa',
        'European fire salamander, Salamandra salamandra',
        'common newt, Triturus vulgaris',
        'eft',
        'spotted salamander, Ambystoma maculatum',
        'axolotl, mud puppy, Ambystoma mexicanum',
        'bullfrog, Rana catesbeiana',
        'tree frog, tree-frog',
        'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
        'loggerhead, loggerhead turtle, Caretta caretta',
        'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',  # noqa: E501
        'mud turtle',
        'terrapin',
        'box turtle, box tortoise',
        'banded gecko',
        'common iguana, iguana, Iguana iguana',
        'American chameleon, anole, Anolis carolinensis',
        'whiptail, whiptail lizard',
        'agama',
        'frilled lizard, Chlamydosaurus kingi',
        'alligator lizard',
        'Gila monster, Heloderma suspectum',
        'green lizard, Lacerta viridis',
        'African chameleon, Chamaeleo chamaeleon',
        'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',  # noqa: E501
        'African crocodile, Nile crocodile, Crocodylus niloticus',
        'American alligator, Alligator mississipiensis',
        'triceratops',
        'thunder snake, worm snake, Carphophis amoenus',
        'ringneck snake, ring-necked snake, ring snake',
        'hognose snake, puff adder, sand viper',
        'green snake, grass snake',
        'king snake, kingsnake',
        'garter snake, grass snake',
        'water snake',
        'vine snake',
        'night snake, Hypsiglena torquata',
        'boa constrictor, Constrictor constrictor',
        'rock python, rock snake, Python sebae',
        'Indian cobra, Naja naja',
        'green mamba',
        'sea snake',
        'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
        'diamondback, diamondback rattlesnake, Crotalus adamanteus',
        'sidewinder, horned rattlesnake, Crotalus cerastes',
        'trilobite',
        'harvestman, daddy longlegs, Phalangium opilio',
        'scorpion',
        'black and gold garden spider, Argiope aurantia',
        'barn spider, Araneus cavaticus',
        'garden spider, Aranea diademata',
        'black widow, Latrodectus mactans',
        'tarantula',
        'wolf spider, hunting spider',
        'tick',
        'centipede',
        'black grouse',
        'ptarmigan',
        'ruffed grouse, partridge, Bonasa umbellus',
        'prairie chicken, prairie grouse, prairie fowl',
        'peacock',
        'quail',
        'partridge',
        'African grey, African gray, Psittacus erithacus',
        'macaw',
        'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
        'lorikeet',
        'coucal',
        'bee eater',
        'hornbill',
        'hummingbird',
        'jacamar',
        'toucan',
        'drake',
        'red-breasted merganser, Mergus serrator',
        'goose']
    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):
        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            extensions=self.IMG_EXTENSIONS,
            test_mode=test_mode,
            file_client_args=file_client_args)
