# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "GT_Masking_work"
# schema_version = "0.20.0"
# engine_version_created_with = "0.101.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.85.0"], ["Video Mask & Inpainting Library", "1.0.0"]]
# node_types_used = [["Griptape Nodes Library", "LoadVideo"], ["Video Mask & Inpainting Library", "VideoMaskInpaintingAllInOneNode"]]
# is_griptape_provided = false
# is_template = false
# is_internal = false
# creation_date = 2026-09-14T06:22:42.954527Z
# last_modified_date = 2026-09-20T07:18:09.689440Z
# 
# ///

import pickle
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.node_library.library_registry import IconVariant, NodeDeprecationMetadata, NodeMetadata
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import AddParameterToNodeRequest, AlterParameterDetailsRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

async def build_workflow() -> None:
    await GriptapeNodes.ahandle_request(RegisterLibraryFromFileRequest(library_name='Griptape Nodes Library', perform_discovery_if_not_found=True))
    await GriptapeNodes.ahandle_request(RegisterLibraryFromFileRequest(library_name='Video Mask & Inpainting Library', perform_discovery_if_not_found=True))
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_workflow():
        context_manager.push_workflow(file_path=__file__)
    # 1. We've collated all of the unique parameter values into a dictionary so that we do not have to duplicate them.
    #    This minimizes the size of the code, especially for large objects like serialized image files.
    # 2. We're using a prefix so that it's clear which Flow these values are associated with.
    # 3. The values are serialized using pickle, which is a binary format. This makes them harder to read, but makes
    #    them consistently save and load. It allows us to serialize complex objects like custom classes, which otherwise
    #    would be difficult to serialize.
    top_level_unique_values_dict = {'079240c3-9cb0-4230-a01d-960e2e0ffc5c': pickle.loads(b'\x80\x04\x95-\x00\x00\x00\x00\x00\x00\x00\x8c)D:\\AI\\GripTape/inputs/videos/20-3-004.mov\x94.'), '4fe3ced1-0bb6-4a01-a9ef-7d7f9cf80663': pickle.loads(b'\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94.'), 'b76b38a4-f85d-487c-bf26-d9a34b9586eb': pickle.loads(b'\x80\x04\x957\x00\x00\x00\x00\x00\x00\x00\x8c3clean background, seamless texture, ultra realistic\x94.'), '8192415c-ad5a-4a4c-9892-0e4ff81bf893': pickle.loads(b'\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\t1920,1080\x94.'), 'ce8c359f-afdd-473a-b583-b99819f6112e': pickle.loads(b'\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\tcogvideox\x94.'), 'c72c544a-372c-4cbe-9b39-0736d599acb1': pickle.loads(b'\x80\x04\x95F\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.video_url_artifact\x94\x8c\x10VideoUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10VideoUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.video_url_artifact\x94\x8c\x02id\x94\x8c cb204a1c66ba41b784dd45f1644817ba\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\n\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c)D:\\AI\\GripTape\\inputs\\videos\\20-3-004.mov\x94ub.'), '73a33798-46c2-4533-8760-ba3d3430a5e6': pickle.loads(b'\x80\x04\x95\x94\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.video_url_artifact\x94\x8c\x10VideoUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94h\x01\x8c\x0bmodule_name\x94h\x00\x8c\x02id\x94\x8c 09aca41fa8ca42eca17d171e9348c97c\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94(\x8c\x05width\x94M\x80\x07\x8c\x06height\x94M8\x04\x8c\x05codec\x94\x8c\x06prores\x94\x8c\nframe_rate\x94G@7\xf9\xdc\xb5\x11"\x87\x8c\tfile_size\x94Jti\x8f\x00\x8c\x06format\x94\x8c\x03mov\x94\x8c\x0bcolor_space\x94\x8c\x05bt709\x94\x8c\x10duration_seconds\x94G@\x03ZH0\x1ay\xffu\x8c\x04name\x94h\x08\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c\x1c{inputs}/videos/20-3-004.mov\x94ub.'), 'adc771b0-82ea-47d7-b75d-828771379ed8': pickle.loads(b'\x80\x04\x95 \x00\x00\x00\x00\x00\x00\x00\x8c\x1c{inputs}/videos/20-3-004.mov\x94.')}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='VideoMaskInpaintingAllInOneNode', specific_library_name='Video Mask & Inpainting Library', node_name='Mask Inpainting (All-in-One)', metadata={'position': {'x': 1044.8285224333313, 'y': 544.4892851181065}, 'tempId': 'placing-1789366968790-hq4lbi', 'library_node_metadata': {'category': 'VideoMaskInpainting', 'description': 'Complete 6-stage video mask & inpainting pipeline in a single node', 'display_name': 'Mask Inpainting (All-in-One)', 'tags': None, 'icon': 'sparkles', 'color': None, 'group': None, 'deprecation': None, 'is_node_group': None, 'declarations': []}, 'library': 'Video Mask & Inpainting Library', 'node_type': 'VideoMaskInpaintingAllInOneNode', 'showaddparameter': False, 'size': {'width': 628, 'height': 837}}, initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(AlterParameterDetailsRequest(parameter_name='masks_dir', ui_options={'display_name': 'Masks Dir', 'hide': False}, initial_setup=True))
        node1_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='LoadVideo', specific_library_name='Griptape Nodes Library', node_name='20-3-004', metadata={'position': {'x': 143.12128824860469, 'y': 448.26824035780726}, 'tempId': 'filedrop-1789886933633-phky39', 'library_node_metadata': {'category': 'video', 'description': 'Loads video files into your workflow', 'display_name': 'Load Video', 'tags': ['video', 'file', 'load'], 'icon': 'file-video', 'color': None, 'group': 'Input/Output', 'deprecation': None, 'is_node_group': None, 'declarations': []}, 'library': 'Griptape Nodes Library', 'node_type': 'LoadVideo', 'showaddparameter': False, 'size': {'width': 600, 'height': 392}}, resolution='resolved', initial_setup=True))).node_name
        await GriptapeNodes.ahandle_request(CreateConnectionRequest(source_node_name=node1_name, source_parameter_name='path', target_node_name=node0_name, target_parameter_name='input_video', initial_setup=True))
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='input_video', node_name=node0_name, value=top_level_unique_values_dict['079240c3-9cb0-4230-a01d-960e2e0ffc5c'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='instruction', node_name=node0_name, value=top_level_unique_values_dict['4fe3ced1-0bb6-4a01-a9ef-7d7f9cf80663'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='prompt', node_name=node0_name, value=top_level_unique_values_dict['b76b38a4-f85d-487c-bf26-d9a34b9586eb'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='negative_prompt', node_name=node0_name, value=top_level_unique_values_dict['4fe3ced1-0bb6-4a01-a9ef-7d7f9cf80663'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='seed_coords', node_name=node0_name, value=top_level_unique_values_dict['8192415c-ad5a-4a4c-9892-0e4ff81bf893'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='box_coords', node_name=node0_name, value=top_level_unique_values_dict['4fe3ced1-0bb6-4a01-a9ef-7d7f9cf80663'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='frame_range', node_name=node0_name, value=top_level_unique_values_dict['4fe3ced1-0bb6-4a01-a9ef-7d7f9cf80663'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='backend', node_name=node0_name, value=top_level_unique_values_dict['ce8c359f-afdd-473a-b583-b99819f6112e'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output_video', node_name=node0_name, value=top_level_unique_values_dict['4fe3ced1-0bb6-4a01-a9ef-7d7f9cf80663'], initial_setup=True, is_output=False))
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='video', node_name=node1_name, value=top_level_unique_values_dict['c72c544a-372c-4cbe-9b39-0736d599acb1'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='video', node_name=node1_name, value=top_level_unique_values_dict['73a33798-46c2-4533-8760-ba3d3430a5e6'], initial_setup=True, is_output=True))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='path', node_name=node1_name, value=top_level_unique_values_dict['adc771b0-82ea-47d7-b75d-828771379ed8'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='path', node_name=node1_name, value=top_level_unique_values_dict['079240c3-9cb0-4230-a01d-960e2e0ffc5c'], initial_setup=True, is_output=True))
