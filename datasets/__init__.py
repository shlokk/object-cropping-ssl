
from .openimages_dataset import OpenImages
from .openimages_dataset_lineval import OpenImages as OpenImages_lineval
from .openimages_dataset_fullimages import OpenImages as OpenImages_fulldataset
from .openimages_dataset_lineval_indiobjects import OpenImages as OpenImages_lineval_indiobjects
from .openimages_dataset_fullimages_with_random_seperated_crops import OpenImages as OpenImages_random_seperated_crops
from .openimages_dataset_bing import  OpenImages as  OpenImages_Bing

def get_trainval_datasets(tag, resize,args='',rescale_parameter=0,radius=0):

    if tag == 'openimages':
        return OpenImages(phase='train', resize=resize, DATAPATH = args.DATAPATH,args=args)
    elif tag == 'openimages_sup':
        return OpenImages(phase='train_sup', resize=resize,DATAPATH = args.DATAPATH,args=args), OpenImages(phase='val_sup', resize=resize,DATAPATH = args.DATAPATH,args=args)
    elif tag == 'openimages_lin':
        return OpenImages_lineval(phase='train_sup', resize=resize,DATAPATH = args.DATAPATH,args=args), OpenImages_lineval(phase='val_sup', resize=resize,DATAPATH = args.DATAPATH,args=args)
    elif tag == 'bing_crops':
        return OpenImages_Bing(phase='train', resize=resize, full_dataset=False, DATAPATH=args.DATAPATH,
                                                 radius=radius, args=args), OpenImages_Bing(
            phase='val', resize=resize, DATAPATH=args.DATAPATH, radius=radius, args=args)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))
