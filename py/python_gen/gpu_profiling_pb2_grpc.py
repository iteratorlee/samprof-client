# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import gpu_profiling_pb2 as gpu__profiling__pb2


class GPUProfilingServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.PerformGPUProfiling = channel.unary_unary(
                '/gpuprofiling.GPUProfilingService/PerformGPUProfiling',
                request_serializer=gpu__profiling__pb2.GPUProfilingRequest.SerializeToString,
                response_deserializer=gpu__profiling__pb2.GPUProfilingResponse.FromString,
                )


class GPUProfilingServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def PerformGPUProfiling(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GPUProfilingServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'PerformGPUProfiling': grpc.unary_unary_rpc_method_handler(
                    servicer.PerformGPUProfiling,
                    request_deserializer=gpu__profiling__pb2.GPUProfilingRequest.FromString,
                    response_serializer=gpu__profiling__pb2.GPUProfilingResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'gpuprofiling.GPUProfilingService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GPUProfilingService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def PerformGPUProfiling(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gpuprofiling.GPUProfilingService/PerformGPUProfiling',
            gpu__profiling__pb2.GPUProfilingRequest.SerializeToString,
            gpu__profiling__pb2.GPUProfilingResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
