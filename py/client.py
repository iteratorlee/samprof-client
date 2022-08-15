import python_gen as gpuprofiling
import argparse
import grpc
import logging
from prune import pruneCCT
def getProfilingResponseFromPB(s:str)->gpuprofiling.GPUProfilingResponse:
    with open(s, "rb") as f:
        return gpuprofiling.GPUProfilingResponse.FromString(f.read())
def getProfilingResponseFromFile(s:str)->gpuprofiling.GPUProfilingResponse:
    return gpuprofiling.GPUProfilingResponse()
if __name__ =="__main__":
    logging.basicConfig(level=logging.INFO)
    parser=argparse.ArgumentParser()
    parser.add_argument('--addr',type=str,dest='addr',default='localhost:8886',help='the address to connect to')
    parser.add_argument('--duration',type=int,dest='duration',default=2000,help='PC sampling duration')
    parser.add_argument('--dumpfn',type=str,dest='dumpfn',help='the dumped profile filename')
    parser.add_argument('--pbfn',type=str,dest='pbfn',help='the dumped pb profile filename')
    parser.add_argument('--prune',dest='prune',action='store_true',help='prune the cpuCCTS')
    args=parser.parse_args()
    r=gpuprofiling.GPUProfilingResponse()
    if args.pbfn is not None:
        logging.info(f'loading response from pre-dumped pb: {args.pbfn}')
        r=getProfilingResponseFromPB(args.pbfn)
    elif args.dumpfn is not None:
        logging.info(f'loading response from pre-dumped txt: {args.dumpfn}')
        r=getProfilingResponseFromFile(args.dumpfn)
    else:
        logging.info('no dumped file, issuing rpc request')
        with grpc.insecure_channel(args.addr) as channel:
            stub = gpuprofiling.GPUProfilingServiceStub(channel)
            r=stub.PerformGPUProfiling(gpuprofiling.GPUProfilingRequest(duration=args.duration))
    if args.prune is not None:
        for cpuCCT in r.cpuCallingCtxTree:
            pruneCCT(cpuCCT)

    #/root/GVProf-samples/pytorch/profiling_response.pb.gz
    #print(r)
