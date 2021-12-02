
from .openimages_dataset import OpenImages
from .openimages_dataset_lineval import OpenImages as OpenImages_lineval
from .openimages_dataset_fullimages import OpenImages as OpenImages_fulldataset
from .openimages_dataset_lineval_indiobjects import OpenImages as OpenImages_lineval_indiobjects
from .openimages_dataset_fullimages_with_random_seperated_crops import OpenImages as OpenImages_random_seperated_crops
from .openimages_dataset_bing import  OpenImages as  OpenImages_Bing
DATAPATH = '/home/dilipkay_google_com/filestore/shlok/datasets/openimages/all_images/'

def get_trainval_datasets(tag, resize,args='',rescale_parameter=0,radius=0):
    # print(args)

    if tag == 'openimages':
        return OpenImages(phase='train', resize=resize,DATAPATH=DATAPATH,args=args)
    elif tag == 'openimages_sup':
        return OpenImages(phase='train_sup', resize=resize,DATAPATH=DATAPATH,args=args), OpenImages(phase='val_sup', resize=resize,DATAPATH=DATAPATH,args=args)
    elif tag == 'openimages_lin':
        return OpenImages_lineval(phase='train_sup', resize=resize,DATAPATH=DATAPATH,args=args), OpenImages_lineval(phase='val_sup', resize=resize,DATAPATH=DATAPATH,args=args)
    elif tag == 'openimages_full_dataset':
        print(args)
        return OpenImages_fulldataset(phase='train', resize=resize,DATAPATH=DATAPATH,radius=radius,args=args), OpenImages_fulldataset(phase='train', resize=resize,DATAPATH=DATAPATH,radius=radius,args=args)
    elif tag == 'openimages_full_dataset_indivual_object':
        return OpenImages_fulldataset(phase='train', resize=resize,full_dataset=False,DATAPATH=DATAPATH,args=args), OpenImages_fulldataset(phase='val', resize=resize,DATAPATH=DATAPATH,args=args)
    elif tag == 'openimages_rescale_crops_before':
        return OpenImages_fulldataset(phase='train',resize=resize,full_dataset=False,rescale_crops_before=True,DATAPATH=DATAPATH,rescale_parameter=rescale_parameter,args=args)\
            ,OpenImages_fulldataset(phase='val',resize=resize,full_dataset=False,rescale_crops_before=True,DATAPATH=DATAPATH,rescale_parameter=rescale_parameter,args=args)
    elif tag == 'openimages_lin_indivual_object':
        return OpenImages_lineval_indiobjects(phase='train', resize=resize,full_dataset=True,DATAPATH=DATAPATH,args=args), OpenImages_lineval_indiobjects(phase='val_sup', resize=resize,DATAPATH=DATAPATH,args=args) #have made change from tran_sup
    elif tag == 'openimages_random_seperated_crops':
        return OpenImages_random_seperated_crops(phase='train', resize=resize,full_dataset=False,DATAPATH=DATAPATH,radius=radius,args=args), OpenImages_random_seperated_crops(phase='val', resize=resize,DATAPATH=DATAPATH,radius=radius,args=args)
    elif tag == 'bing_crops':
        return OpenImages_Bing(phase='train', resize=resize, full_dataset=False, DATAPATH=DATAPATH,
                                                 radius=radius, args=args), OpenImages_Bing(
            phase='val', resize=resize, DATAPATH=DATAPATH, radius=radius, args=args)

    else:
        raise ValueError('Unsupported Tag {}'.format(tag))
