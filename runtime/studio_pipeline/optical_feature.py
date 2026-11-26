"""Compare one source-located focal feature when an adjacent edge dominates a detail crop."""
import json
from PIL import Image
from evidence import ask


def unresolved_neighbor(check):
    loupe=check.get('contour_magnification') or {}
    orders=[loupe.get(key) or {} for key in ('forward','reverse')]
    attempt=check.get('subject_optics')
    untested=(attempt is None or (attempt.get('orders')==[] and 'search' not in attempt))
    return (not check.get('confirmed') and untested
            and check.get('trace',{}).get('new_artifact') is False
            and bool(loupe.get('line_pair'))
            and all(v.get('kind')=='contrast_or_texture_only' for v in orders)
            and [v.get('finer_version') for v in orders]==['second','first'])


def source_loupe(low,high,label,feature):
    from masks import focal_subject_mask
    from print_artifacts import resolved_line_pair
    if low.size!=high.size or min(low.size)<96:return None
    mask,box=focal_subject_mask(low,Image.new('L',low.size,255),label,feature,require_feature=True)
    if not box:return None
    # Include a margin around narrow features so a pair of close marks is measurable.
    x,y,r,b=box
    pad_x=max(12,(100-r+x+1)//2);pad_y=max(12,(100-b+y+1)//2)
    extent=(max(0,x-pad_x),max(0,y-pad_y),min(low.width,r+pad_x),min(low.height,b+pad_y))
    measured=resolved_line_pair(low.crop(extent),high.crop(extent),centers=mask.crop(extent))
    if measured:
        cx,cy=measured['center'];cx+=extent[0];cy+=extent[1]
        witness={'line_pair':measured}
    else:
        # Curved contours such as an iris can be fully resolved without two
        # parallel strokes. Choose a visible turn on the lower-scale source.
        # Its position is only a locator. Both blinded readers still judge it.
        import cv2
        import numpy as np
        from print_judge import resize_only
        if resize_only((low,high))['condition']=='resize_or_sharpen':return None
        ink=np.array(low.convert('L'),dtype=np.float32)/255
        corners=cv2.cornerMinEigenVal(ink,blockSize=7,ksize=3)
        corners[np.asarray(mask)<128]=0
        _,strength,_,(cx,cy)=cv2.minMaxLoc(corners)
        if strength<.00015:return None
        witness={'source_contour_turn':[cx,cy]}

    if not (x<=cx<r and y<=cy<b and mask.getpixel((cx,cy))>=128):return None
    x=max(0,min(low.width-96,cx-48));y=max(0,min(low.height-96,cy-48))
    return dict(feature_box=list(box),pixels_in_field=[x,y,x+96,y+96],**witness)


def verify_source_feature(url,low,high,overview,models):
    from focal_edit import describe_patch
    desc=describe_patch(url,low,overview,localize=True)
    result=dict(confirmed=False,source_inventory=desc,orders=[],search='inside_localized_feature')
    if desc.get('scope')!='bounded_subject' or not desc.get('label') or not desc.get('feature'):return result
    site=source_loupe(low,high,desc['label'],desc['feature'])
    if not site:return result
    result.update(site)
    crops=[im.crop(site['pixels_in_field']).resize((288,288),Image.Resampling.NEAREST) for im in (low,high)]
    fields={'located':{'type':'boolean'},'finer':{'enum':['first','second','equivalent','unreadable']},
            'improvement':{'enum':['resolved_shape','contrast_or_noise','none','artifact']},
            'trace':{'type':'string'},'new_artifact':{'type':'boolean'}}
    schema={'type':'object','properties':fields,'required':list(fields)}
    cue=('Compare these two anonymous enlargements at exactly the same optical scale. On the original '
         'ungraded print the independently localized feature was '+desc['feature']+' on the '+desc['label']+
         '. First check it is actually present here. Follow one shape inside that feature through a turn '
         'or separation of adjacent marks. Which print resolves that shape more clearly? Name the '
         'difference in the drawing itself. A darker edge, rougher surface, pixel halo or noise is insufficient. '
         'If the crop shows neighboring scenery or the intended feature cannot be identified, mark located false.')
    if len(models)!=2 or len(set(models))!=2:return result
    for order,tag,candidates in zip(('second','first'),models,(crops,crops[::-1])):
        row=json.loads(ask(url,candidates,cue,schema,tag=tag))
        result['orders'].append(dict(model=tag,**row))
        if (row.get('located') is not True or row.get('finer')!=order or
                row.get('improvement')!='resolved_shape' or row.get('new_artifact') is not False or
                not isinstance(row.get('trace'),str) or len(row['trace'].split())<4):return result
    result['confirmed']=True
    return result
