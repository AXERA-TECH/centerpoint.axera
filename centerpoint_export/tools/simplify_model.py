#!/usr/bin/env python
# coding: utf-8

import onnx
from onnx import numpy_helper, helper
import numpy as np
from onnxsim import simplify

def matmul_to_conv2d(node, init_dict):
    weight_name = node.input[1]
    weight_tensor = init_dict[weight_name]
    weight = numpy_helper.to_array(weight_tensor)
    if len(weight.shape) == 2:
        # Transpose (K, C) -> (C, K) then expand to (C, K, 1, 1) or similar depending on Conv kernel
        # Standard PointPillars MatMul usually needs weights as (Out, In, 1, 1) for Conv 1x1
        weight = np.expand_dims(weight.transpose(1,0),[2,3])
    weight_tensor = numpy_helper.from_array(weight, name=weight_name)
    init_dict[weight_name] = weight_tensor

def delete_init(model):
    # Only clear initializers that are strictly used by deleted nodes? 
    # Current logic clears all and restores from init_dict, which is safe IF init_dict is complete.
    init_len = len(model.graph.initializer)
    for i in range(init_len):
        model.graph.initializer.pop()

def convert_input_nhwc_nchw(model):
    batch_dim = 1
    # Check if input exists
    if not model.graph.input:
        return ""
        
    dim_list = [dim_val.dim_value for dim_val in model.graph.input[0].type.tensor_type.shape.dim]
    old_input_name = model.graph.input[0].name
    
    # Handle scalar or weird shapes gracefully
    if len(dim_list) >= 1:
        dim_list.insert(0, batch_dim)
        # Ensure we have 4 dims for NHWC->NCHW permute
        if len(dim_list) == 4:
            dim_list = np.array(dim_list)[[0,3,1,2]]
    
    input_node = onnx.helper.make_tensor_value_info('input.1', \
                                                    onnx.TensorProto.FLOAT, dim_list.tolist())
    model.graph.input.pop()
    model.graph.input.append(input_node)
    
    # Fix output shape similarly
    if model.graph.output:
        dim_list = [dim_val.dim_value for dim_val in model.graph.output[0].type.tensor_type.shape.dim]
        dim_list.insert(0, batch_dim)
        dim_list.insert(2, 1)
        if len(dim_list) == 4:
            dim_list = np.array(dim_list)[[0,3,1,2]]
    
        out_node = onnx.helper.make_tensor_value_info(model.graph.output[0].name, \
                                                      onnx.TensorProto.FLOAT, dim_list.tolist())
        model.graph.output.pop()
        model.graph.output.append(out_node)
        
    return old_input_name

def reducemax_to_maxpool(node):
    # In-place modification to preserve topological order
    node.op_type = "MaxPool"
    if node.attribute:
        del node.attribute[:]
    
    node.attribute.extend([
        helper.make_attribute("ceil_mode", 0),
        helper.make_attribute("kernel_shape", [1, 20]),
        helper.make_attribute("pads", [0, 0, 0, 0]),
        helper.make_attribute("strides", [1, 1])
    ])

def convert_tile(node, init_dict):
    arr_name = node.input[1]
    arr = np.array([1,1,1,20], np.int64)
    tensor = numpy_helper.from_array(arr, name=arr_name)
    init_dict[arr_name] = tensor

def simplify_model(model_path):
    model = onnx.load(model_path)
    if model is None:
        print("File %s is not find! "%model_path)
    return simplify(model)

def simplify_pfe_rpn_model(pfe_model_path, rpn_model_path):
    # Initial simplify is fine, it cleans up the raw export
    print(f"Simplifying {pfe_model_path}...")
    model, check = simplify_model(pfe_model_path)
    if not check:
        print("[ERROR]:Simplify %s error!"% pfe_model_path)    
    onnx.save(model, pfe_model_path)

    print(f"Simplifying {rpn_model_path}...")
    model, check = simplify_model(rpn_model_path)
    if not check:
        print("[ERROR]:Simplify %s error!"% rpn_model_path)    
    onnx.save(model, rpn_model_path)

def get_real_input(input_name, delete_dict):
    curr_name = input_name
    # Trace back through deleted nodes until we find a living node
    while curr_name in delete_dict:
        curr_name = delete_dict[curr_name].input[0]
    return curr_name

if __name__ == "__main__":
    pfe_model_path = "./onnx_model/pfe.onnx"
    pfe_model_save_path = "./onnx_model/pfe_sim.onnx"
    rpn_model_path = "./onnx_model/rpn.onnx"
  
    # 1. Pre-simplify raw models
    simplify_pfe_rpn_model(pfe_model_path, rpn_model_path)
    
    # 2. Load PFE for hacking
    model = onnx.load(pfe_model_path)
    init_dict = {}
    for init_node in model.graph.initializer:
        init_dict[init_node.name] = init_node
    
    # Identify nodes to delete (Transpose/Expand/Squeeze)
    delete_dict = {}
    for node in model.graph.node:
        if node.op_type in {"Transpose", "Expand", "Squeeze"}:
            # Map output name to the node itself
            delete_dict[node.output[0]] = node

    # Clear ValueInfo and Initializers (will restore Initializers later)
    if model.graph.value_info:
        del model.graph.value_info[:]
    delete_init(model)

    old_input_name = convert_input_nhwc_nchw(model)
    
    # 3. Main Modification Loop (In-place)
    # We iterate over a copy of the list so we can modify inputs safely
    # BUT we do NOT remove nodes yet to allow get_real_input to work
    for node in model.graph.node:
        
        # Skip processing nodes that are marked for deletion
        if node.output[0] in delete_dict:
            continue

        if node.op_type == "MatMul":
            node.op_type = "Conv"
            matmul_to_conv2d(node, init_dict)
        
        # Rewire inputs
        for i, input_name in enumerate(node.input):
            target_input = get_real_input(input_name, delete_dict)
            
            if target_input == old_input_name:
                target_input = 'input.1'
                
            if target_input != input_name:
                node.input[i] = target_input
        
        if node.op_type == "ReduceMax":
            reducemax_to_maxpool(node)
            
        if node.op_type == "Tile":
            convert_tile(node, init_dict)
            
        if node.op_type == "Concat":
            # Concat axis usually needs adjustment if we changed layout
            # Standard PointPillars Concat is usually on axis 1 (C)
            if node.attribute:
                node.attribute[0].i = 1

    # 4. Handle Graph Outputs
    for node in model.graph.output:
        if node.name in delete_dict:
            node.name = get_real_input(node.name, delete_dict)

    # 5. Restore Initializers
    for name, tensor in init_dict.items():
        model.graph.initializer.append(tensor)
        
    # 6. Actually Remove Deleted Nodes
    # Construct a new node list excluding deleted ones
    new_nodes = []
    for node in model.graph.node:
        if node.output[0] not in delete_dict:
            new_nodes.append(node)
    
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # [IMPORTANT] Do NOT run simplify here. 
    # It renames tensors and might hide 'rpn_input'.
    # Since we modified in-place, the topological order is preserved.
    
    print(f"Saving modified model to {pfe_model_save_path}...")
    onnx.save(model, pfe_model_save_path)
    print("Done. Please try compiling this version.")