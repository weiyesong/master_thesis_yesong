from __future__ import annotations
import csv, hashlib, json, os, shutil, time, zipfile, resource
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path('/workspace')
OUT=Path(__file__).resolve().parents[1]
PARAMS=json.loads((OUT/'parameters.json').read_text())
ARCHIVE=pq.read_table(ROOT/'research_data/manifest.parquet').to_pylist()
MASTER=list(csv.DictReader((ROOT/'reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv').open()))
OFF=list(csv.DictReader((ROOT/'reports/core_rq_completion_20260921/mc_dropout_three_way.csv').open()))


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');os.replace(tmp,path)


def csvout(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def arrhash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def key(r):
    return tuple(str(r[k]) for k in ['dataset','model','adaptation','seed'])


def slug(r):
    return '__'.join(str(r[k]) for k in ['dataset','model','adaptation','method','seed']).rstrip('_')


def archive_match(r,role):
    xs=[x for x in ARCHIVE if x['role']==role and all(str(x[k])==str(r[k]) for k in ['dataset','model','adaptation'])
        and (x['seed'] is None or str(x['seed'])==str(r['seed']))]
    if len(xs)!=1:raise ValueError((role,r,len(xs)))
    return xs[0]


def archive_path(r,role):
    return ROOT/'research_data'/archive_match(r,role)['archive_path']


def expected_row(r):
    if r['method']=='mc_dropout_off':
        x=next(x for x in OFF if key(x)==key(r))
        return {k[4:]:v for k,v in x.items() if k.startswith('Off_')},x
    rs=[x for x in MASTER if x['record_type'] in ['INDIVIDUAL_SEED','ENSEMBLE'] and x['uq_method']==r['method']
        and all(str(x[k])==str(r[k]) for k in ['dataset','model','adaptation'])
        and (x['record_type']=='ENSEMBLE' or x['seed']==str(r['seed']))]
    if len(rs)!=1:raise ValueError((r,rs))
    return rs[0],rs[0]


def build_inputs():
    rows=list(csv.DictReader((ROOT/'reports/thesis_followup_design_20260926/planned_A_prediction_objects.csv').open()))
    out=[]
    for r in rows:
        p=ROOT/r['representative_prediction_path'];expect,source=expected_row(r)
        raw=''
        if r['method']=='mc_dropout':raw=str(archive_path(r,'classification_mc_raw_passes' if r['task']=='classification' else 'segmentation_mc_raw_subset').relative_to(ROOT))
        if r['method']=='deep_ensemble':raw=str(archive_path(r,'classification_ensemble_members' if r['task']=='classification' else 'segmentation_ensemble_member_subset').relative_to(ROOT))
        a=next((x for x in ARCHIVE if 'research_data/'+x['archive_path']==str(p.relative_to(ROOT))),None)
        digest=a['sha256'] if a else sha(p)
        out.append({**r,'object_id':slug(r),'prediction_path':str(p.relative_to(ROOT)),'raw_path':raw,
                    'checkpoint_path':source.get('checkpoint_path',''),'run_id':source.get('run_id',''),
                    'sha256':digest,'hash_basis':'archive_manifest_expected_hash; bytes checked' if a else 'computed_this_execution',
                    'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'split':'test',
                    'analysis_scope':'classification_full' if r['task']=='classification' else 'full_image_and_fixed32_pixel',
                    'decomposition_scope':'full' if r['method']=='mc_dropout' or (r['method']=='deep_ensemble' and r['task']=='classification') else ('fixed32_only' if r['method']=='deep_ensemble' else 'NA_single_distribution'),
                    'status':'READY','missing_reason':''})
    csvout(OUT/'inputs_manifest.csv',out)
    return out


def get_inputs():
    return list(csv.DictReader((OUT/'inputs_manifest.csv').open()))


def build_groups():
    allrows=[];summary={}
    for ds in ['eurosat','treesatai','cloudsen12','spacenet7']:
        path=(ROOT/'splits/eurosat_70_10_10_10_spatial20m/eurosat_splits.csv' if ds=='eurosat' else ROOT/f'reports/dataset_manifests/{ds}_actual_manifest.csv')
        df=pd.read_csv(path); test=df[df.split=='test'].copy()
        if ds=='eurosat':test['group_id']=test.spatial_group.astype(str);basis='provided spatial_group'
        elif ds=='treesatai':test['group_id']=test.sample_id;basis='image; all test coordinates distinct; nearby correlation unknown'
        elif ds=='spacenet7':test['group_id']=test.aoi;basis='AOI'
        else:
            parent=list(range(len(df)))
            def find(x):
                while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
                return x
            for col in ['roi_id','equi_id','sentinel2_product_id']:
                seen={}
                for i,v in enumerate(df[col]):
                    if pd.isna(v):continue
                    if v in seen:parent[find(i)]=find(seen[v])
                    else:seen[v]=i
            test['group_id']=['source_cc_'+str(find(i)) for i in test.index];basis='full-manifest roi/equi/product connected components'
        for _,r in test.iterrows():allrows.append({'dataset':ds,'sample_id':str(r.sample_id),'group_id':str(r.group_id),'group_basis':basis,'source_path':str(path.relative_to(ROOT))})
        sizes=test.groupby('group_id').size()
        summary[ds]={'images':len(test),'groups':len(sizes),'minimum_group_images':int(sizes.min()),'maximum_group_images':int(sizes.max()),'basis':basis,'source_sha256':sha(path)}
    csvout(OUT/'sample_groups.csv',allrows);dump(OUT/'grouping_summary.json',summary)


def groups_for(ds,ids):
    d=pd.read_csv(OUT/'sample_groups.csv',dtype=str);d=d[d.dataset==ds].set_index('sample_id')
    if set(ids)!=set(d.index):raise ValueError(('group IDs differ',ds))
    return d.loc[ids].group_id.to_numpy()


class LazyNPZ:
    """Decompress numeric arrays once to execution-owned NPY scratch memmaps."""
    def __init__(self,path,tag):
        self.path=Path(path);self.z=zipfile.ZipFile(path);self.dir=OUT/'scratch'/tag
        self.dir.mkdir(parents=True,exist_ok=True);self.cache={};self.seconds=0.;self.uncompressed_bytes=0
        self.files=[p[:-4] for p in self.z.namelist() if p.endswith('.npy')]
    def __contains__(self,k):return k in self.files
    def __getitem__(self,k):
        if k not in self.cache:
            t=time.perf_counter();p=self.dir/(k+'.npy')
            with self.z.open(k+'.npy') as src,p.open('wb') as dst:shutil.copyfileobj(src,dst,length=8*1024*1024)
            self.uncompressed_bytes+=p.stat().st_size
            try:self.cache[k]=np.load(p,mmap_mode='r',allow_pickle=False)
            except ValueError as e:
                if 'Python objects' not in str(e):raise
                # Legacy local ensemble IDs are documented trusted object arrays.
                self.cache[k]=np.load(p,allow_pickle=True)
            self.seconds+=time.perf_counter()-t
        return self.cache[k]
    def close(self):
        for a in self.cache.values():
            if hasattr(a,'_mmap') and a._mmap is not None:a._mmap.close()
        self.cache.clear();self.z.close();shutil.rmtree(self.dir)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def rss_gib():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2


def matrix_col(df,name):return np.stack(df[name].to_numpy())


if __name__=='__main__':
    t=time.perf_counter();rows=build_inputs();build_groups()
    dump(OUT/'preparation_summary.json',{'objects':len(rows),'seconds':time.perf_counter()-t,'peak_rss_GiB':rss_gib()})
    print('Prepared',len(rows),'objects and actual metadata groups')
