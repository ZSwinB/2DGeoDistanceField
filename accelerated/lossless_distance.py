"""Lossless, indexed FP32 distance maps; bit-pattern prediction, never float quantization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
from numcodecs import Blosc

MAGIC=b'DMAPLS01'


try:
    from numba import njit

    @njit(cache=True,nogil=True)
    def restore_u32(delta):
        out=np.empty(delta.shape,np.uint32)
        for y in range(len(delta)):
            running=np.uint32(0)
            for x in range(delta.shape[1]):
                running=np.uint32(running+delta[y,x])
                out[y,x]=np.uint32(running+(out[y-1,x] if y else np.uint32(0)))
        return out
except ImportError:
    def restore_u32(delta):
        return np.cumsum(np.cumsum(delta,axis=1,dtype=np.uint32),axis=0,dtype=np.uint32)


def predict_u32(a):
    values=np.ascontiguousarray(a).view(np.uint32)
    first=values.copy();first[:,1:]=values[:,1:]-values[:,:-1]
    out=first.copy();out[1:]=first[1:]-first[:-1]
    return out


def write_distance(path,a,chunk_size=256,clevel=3,metadata=None):
    if a.ndim!=2 or a.dtype!=np.dtype('<f4') or not np.little_endian:
        raise TypeError('Expected a 2D little-endian FP32 field; dtype conversion is not performed')
    if min(a.shape)<1 or chunk_size<1:raise ValueError('Empty field or invalid chunk size')
    codec=Blosc(cname='zstd',clevel=clevel,shuffle=Blosc.BITSHUFFLE)
    records=[];payload=[];offset=0
    for y in range(0,a.shape[0],chunk_size):
        for x in range(0,a.shape[1],chunk_size):
            tile=np.ascontiguousarray(a[y:y+chunk_size,x:x+chunk_size]);data=codec.encode(predict_u32(tile))
            records.append({'y':y,'x':x,'shape':list(tile.shape),'offset':offset,'size':len(data),'sha256':hashlib.sha256(tile.tobytes()).hexdigest()})
            payload.append(data);offset+=len(data)
    header={'version':1,'shape':list(a.shape),'dtype':'<f4','transform':'u32_delta2d_mod_2pow32','codec':codec.get_config(),
            'chunk_size':chunk_size,'chunks':records,'metadata':metadata or {}}
    encoded=json.dumps(header,separators=(',',':')).encode('utf8')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            tmp=Path(stream.name);stream.write(MAGIC);stream.write(struct.pack('<Q',len(encoded)))
            stream.write(hashlib.sha256(encoded).digest());stream.write(encoded)
            for block in payload:stream.write(block)
        os.replace(tmp,path)
    finally:
        if tmp is not None and tmp.exists():tmp.unlink()
    return header


class DistanceReader:
    def __init__(self,path):
        self.path=Path(path)
        with self.path.open('rb') as stream:
            if stream.read(8)!=MAGIC:raise ValueError('Not a DMAPLS01 distance map')
            header_size=struct.unpack('<Q',stream.read(8))[0]
            if header_size>min(64*1024**2,self.path.stat().st_size-48):raise ValueError('Invalid distance-map header size')
            digest=stream.read(32);encoded=stream.read(header_size)
            if hashlib.sha256(encoded).digest()!=digest:raise ValueError('Distance-map header checksum mismatch')
            self.header=json.loads(encoded);self.payload_offset=48+header_size
        if self.header['version']!=1 or self.header['dtype']!='<f4' or self.header['transform']!='u32_delta2d_mod_2pow32':
            raise ValueError('Unsupported distance-map encoding')
        config=dict(self.header['codec'])
        if config.pop('id',None)!='blosc':raise ValueError('Unsupported distance-map compressor')
        self.shape=tuple(self.header['shape']);self.codec=Blosc.from_config(config);self.chunks_decoded=0

    def read_window(self,y0,y1,x0,x1,verify=True):
        if not (0<=y0<=y1<=self.shape[0] and 0<=x0<=x1<=self.shape[1]):raise ValueError('Window outside field')
        out=np.empty((y1-y0,x1-x0),np.float32);self.chunks_decoded=0
        with self.path.open('rb') as stream:
            for block in self.header['chunks']:
                y,x=block['y'],block['x'];h,w=block['shape']
                ya,yb=max(y0,y),min(y1,y+h);xa,xb=max(x0,x),min(x1,x+w)
                if ya>=yb or xa>=xb:continue
                stream.seek(self.payload_offset+block['offset']);data=stream.read(block['size'])
                if len(data)!=block['size']:raise ValueError('Truncated distance-map block')
                delta=np.frombuffer(self.codec.decode(data),np.uint32).reshape(h,w)
                tile=restore_u32(delta).view(np.float32)
                if verify and hashlib.sha256(tile.tobytes()).hexdigest()!=block['sha256']:raise ValueError('Distance-map block checksum mismatch')
                out[ya-y0:yb-y0,xa-x0:xb-x0]=tile[ya-y:yb-y,xa-x:xb-x];self.chunks_decoded+=1
        return out

    def read(self,verify=True):return self.read_window(0,self.shape[0],0,self.shape[1],verify)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    pack=sub.add_parser('pack');pack.add_argument('source');pack.add_argument('destination');pack.add_argument('--chunk-size',type=int,default=256)
    unpack=sub.add_parser('unpack');unpack.add_argument('source');unpack.add_argument('destination')
    info=sub.add_parser('info');info.add_argument('source')
    args=parser.parse_args()
    if args.action=='pack':write_distance(args.destination,np.load(args.source,allow_pickle=False),args.chunk_size)
    elif args.action=='unpack':np.save(args.destination,DistanceReader(args.source).read())
    else:print(json.dumps(DistanceReader(args.source).header,indent=2))
if __name__=='__main__':main()
