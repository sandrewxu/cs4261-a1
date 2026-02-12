import math
import torch
import torch.nn.functional as F
from starter.gpt_with_kv_mha import GPTModel

config = {
    "vocab_size": 64,
    "context_length": 8,
    "emb_dim": 32,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False
}
batch_size, seq_len = 2, 8

def trace_forward_pass():
    """
    Demonstrates grad_fn creation during forward pass.
    """
    print("="*50)
    print("Forward Pass with grad_fn Tracking")
    print("="*50)

    model = GPTModel(config)
    model.train()

    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input requires_grad: {input_ids.requires_grad}")  # False (leaf, not float)

    logits = model(input_ids)
    print(f"\nOutput (logits) shape: {logits.shape}")
    print(f"Output grad_fn: {logits.grad_fn}")  # Will show grad_fn chain
    print(f"Output requires_grad: {logits.requires_grad}")  # True

    # Trace through the computation graph
    print("\n" + "-"*50)
    print("Tracing backward through grad_fn chain:")
    print("-"*50)

    # Topology search (DFS)
    visited = set()

    def print_grad_fn(grad_fn, depth=0, max_depth=10):
        """Recursively print grad_fn tree structure."""
        if grad_fn is None or depth >= max_depth or id(grad_fn) in visited:
            return
        
        visited.add(id(grad_fn))
        indent = "  " * depth

        # Print current grad_fn with more details
        grad_fn_name = type(grad_fn).__name__
        print(f"{indent}└> {grad_fn_name}")

        # Recursively print next functions
        if hasattr(grad_fn, 'next_functions'):
            next_fns = grad_fn.next_functions
            for i, (next_grad_fn, _) in enumerate(next_fns):
                if next_grad_fn is not None:
                    if i > 0 and depth < max_depth - 1:  # Show branches
                        print(f"{indent}  ├> [branch {i}]")
                    print_grad_fn(next_grad_fn, depth + 1, max_depth)
    
    print_grad_fn(logits.grad_fn)

    return model, logits

def visualize_graph_with_torchviz(model, logits):
    """
    Visualize computation graph using torchviz.
    """
    try:
        from torchviz import make_dot

        # Compute a scalar loss for visualization
        target = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        loss = F.cross_entropy(logits.view(-1, config["vocab_size"]), target.view(-1))

        # Generate visualization
        dot = make_dot(loss, params=dict(model.named_parameters()))

        # Save the graph
        output_file = "computation_graph"
        dot.render(output_file, format='png', cleanup=True)

        print(f"\nVisualization saved to: {output_file}.png")
    except Exception as e:
        print(f"\nError generating visualization: {e}")

def main():
    model, logits = trace_forward_pass()
    visualize_graph_with_torchviz(model, logits)

if __name__ == "__main__":
    main()
