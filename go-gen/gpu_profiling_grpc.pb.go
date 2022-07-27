// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.2.0
// - protoc             v3.19.4
// source: gpu_profiling.proto

package gpuprofiling

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

// GPUProfilingServiceClient is the client API for GPUProfilingService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type GPUProfilingServiceClient interface {
	PerformGPUProfiling(ctx context.Context, in *GPUProfilingRequest, opts ...grpc.CallOption) (*GPUProfilingResponse, error)
}

type gPUProfilingServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewGPUProfilingServiceClient(cc grpc.ClientConnInterface) GPUProfilingServiceClient {
	return &gPUProfilingServiceClient{cc}
}

func (c *gPUProfilingServiceClient) PerformGPUProfiling(ctx context.Context, in *GPUProfilingRequest, opts ...grpc.CallOption) (*GPUProfilingResponse, error) {
	out := new(GPUProfilingResponse)
	err := c.cc.Invoke(ctx, "/gpuprofiling.GPUProfilingService/PerformGPUProfiling", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// GPUProfilingServiceServer is the server API for GPUProfilingService service.
// All implementations must embed UnimplementedGPUProfilingServiceServer
// for forward compatibility
type GPUProfilingServiceServer interface {
	PerformGPUProfiling(context.Context, *GPUProfilingRequest) (*GPUProfilingResponse, error)
	mustEmbedUnimplementedGPUProfilingServiceServer()
}

// UnimplementedGPUProfilingServiceServer must be embedded to have forward compatible implementations.
type UnimplementedGPUProfilingServiceServer struct {
}

func (UnimplementedGPUProfilingServiceServer) PerformGPUProfiling(context.Context, *GPUProfilingRequest) (*GPUProfilingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method PerformGPUProfiling not implemented")
}
func (UnimplementedGPUProfilingServiceServer) mustEmbedUnimplementedGPUProfilingServiceServer() {}

// UnsafeGPUProfilingServiceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to GPUProfilingServiceServer will
// result in compilation errors.
type UnsafeGPUProfilingServiceServer interface {
	mustEmbedUnimplementedGPUProfilingServiceServer()
}

func RegisterGPUProfilingServiceServer(s grpc.ServiceRegistrar, srv GPUProfilingServiceServer) {
	s.RegisterService(&GPUProfilingService_ServiceDesc, srv)
}

func _GPUProfilingService_PerformGPUProfiling_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GPUProfilingRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(GPUProfilingServiceServer).PerformGPUProfiling(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/gpuprofiling.GPUProfilingService/PerformGPUProfiling",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(GPUProfilingServiceServer).PerformGPUProfiling(ctx, req.(*GPUProfilingRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// GPUProfilingService_ServiceDesc is the grpc.ServiceDesc for GPUProfilingService service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var GPUProfilingService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "gpuprofiling.GPUProfilingService",
	HandlerType: (*GPUProfilingServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "PerformGPUProfiling",
			Handler:    _GPUProfilingService_PerformGPUProfiling_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "gpu_profiling.proto",
}
