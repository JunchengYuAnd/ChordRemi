import torch
import argparse
from transformers import AutoModelForCausalLM, GenerationConfig
from hello_remi import ChordREMI

# Setup command line argument parser
parser = argparse.ArgumentParser(description='Generate music with specified chord progression')
parser.add_argument('--chords', nargs='+', default=[], help='List of chords in format root_type (e.g., C_maj D_min G_maj)')
parser.add_argument('--output', type=str, default="output.mid", help='Output MIDI filename')
parser.add_argument('--temperature', type=float, default=0.9, help='Temperature for generation (0.1-1.0)')
parser.add_argument('--max_tokens', type=int, default=1024, help='Maximum number of tokens to generate')
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    torch.set_default_device(device)

model = AutoModelForCausalLM.from_pretrained("Models/checkpoint-6200")
tokenizer = ChordREMI(params="tokenizer_exp_02.json")

print("Tokenizer vocabulary loaded")

self_input_tokens = [3]  # Start token

# id list to pytorch tensor and add batch dimension 
input_tensor = torch.tensor(self_input_tokens, dtype=torch.long).unsqueeze(0) 

input_tensor = input_tensor.to(device)

from typing import List

def create_token_control_fn_with_4th_token(
    trigger_token_id: int,
    forced_token_list: List[List[int]],
    force_4th_token: int,
    vocab_size: int
):
    state = {
        'count': 0,        # Number of times trigger_token_id is encountered
        'force_pos': 0     # Position for forced token generation (0 means no forcing, 1 and 2 for forced mode)
    }

    def control_tokens(batch_id, input_ids):
        current_len = input_ids.size(0)

        # First check if we need to force the 4th token (index 3)
        if current_len == 4:
            return [force_4th_token]

        # Check if we're in forced token generation mode
        if state['force_pos'] > 0:
            forced_tokens = forced_token_list[state['count'] - 1]
            token_to_generate = forced_tokens[state['force_pos'] - 1]

            state['force_pos'] += 1
            if state['force_pos'] > 2:
                state['force_pos'] = 0

            return [token_to_generate]

        # Check if current token is the trigger token
        if input_ids[-1].item() == trigger_token_id:
            if state['count'] < len(forced_token_list):
                state['count'] += 1
                forced_tokens = forced_token_list[state['count'] - 1]
                state['force_pos'] = 2  # Force next token as the 2nd token
                
                return [forced_tokens[0]]

        # Allow all tokens for other cases
        return list(range(vocab_size))
    
    return control_tokens

chord_dict = {
    'C_maj': 313,
    'C_min': 314,
    'C_dim': 315,
    'C_aug': 316,
    'C#_maj': 317,
    'C#_min': 318,
    'C#_dim': 319,
    'C#_aug': 320,
    'D_maj': 321,
    'D_min': 322,
    'D_dim': 323,
    'D_aug': 324,
    'D#_maj': 325,
    'D#_min': 326,
    'D#_dim': 327,
    'D#_aug': 328,
    'E_maj': 329,
    'E_min': 330,
    'E_dim': 331,
    'E_aug': 332,
    'F_maj': 333,
    'F_min': 334,
    'F_dim': 335,
    'F_aug': 336,
    'F#_maj': 337,
    'F#_min': 338,
    'F#_dim': 339,
    'F#_aug': 340,
    'G_maj': 341,
    'G_min': 342,
    'G_dim': 343,
    'G_aug': 344,
    'G#_maj': 345,
    'G#_min': 346, 
    'G#_dim': 347,
    'G#_aug': 348,
    'A_maj': 349,
    'A_min': 350,
    'A_dim': 351,
    'A_aug': 352,
    'A#_maj': 353,
    'A#_min': 354,
    'A#_dim': 355,
    'A#_aug': 356,
    'B_maj': 357,
    'B_min': 358,
    'B_dim': 359,
    'B_aug': 360,
    'NoChord': 361
}

# Process the chord progression from command line
def process_chord_progression(chord_inputs):
    if not chord_inputs:
        # Default chord progression if none provided
        return [   
            [chord_dict['D_min'], chord_dict['G_maj']],
            [chord_dict['C_maj'], chord_dict['F_maj']],
            [chord_dict['D_min'], chord_dict['G_maj']],
            [chord_dict['C_maj'], chord_dict['F_maj']]
        ]
    
    # Convert input chords to token pairs
    progression = []
    
    # Process in pairs (or add NoChord if odd number)
    i = 0
    while i < len(chord_inputs):
        if i + 1 < len(chord_inputs):
            # If both chords exist in the dictionary
            if chord_inputs[i] in chord_dict and chord_inputs[i+1] in chord_dict:
                progression.append([chord_dict[chord_inputs[i]], chord_dict[chord_inputs[i+1]]])
            else:
                print(f"Warning: Chord '{chord_inputs[i]}' or '{chord_inputs[i+1]}' not found in dictionary. Using default.")
                progression.append([chord_dict['C_maj'], chord_dict['G_maj']])
        else:
            # If we have an odd number, pair with NoChord
            if chord_inputs[i] in chord_dict:
                progression.append([chord_dict[chord_inputs[i]], chord_dict['NoChord']])
            else:
                print(f"Warning: Chord '{chord_inputs[i]}' not found in dictionary. Using default.")
                progression.append([chord_dict['C_maj'], chord_dict['NoChord']])
        i += 2
    
    return progression

# Get chord progression from command line args
forced_token_list = process_chord_progression(args.chords)

print(f"Using chord progression: {args.chords if args.chords else 'default'}")

trigger_token_id = 3
force_4th_token = 283  
vocab_size = tokenizer.vocab_size

token_control_fn = create_token_control_fn_with_4th_token(
    trigger_token_id, 
    forced_token_list,
    force_4th_token,
    vocab_size
)

generation_config = GenerationConfig( 
    max_new_tokens=args.max_tokens,
    num_beams=1,
    do_sample=True,
    temperature=args.temperature,
    top_k=15,
    top_p=0.90,
    epsilon_cutoff=3e-4,
    eta_cutoff=1e-3,
    pad_token_id=tokenizer.pad_token_id
)

print("Generating music...")
generated_token_ids = model.generate( 
    input_tensor, 
    generation_config=generation_config,
    prefix_allowed_tokens_fn=token_control_fn,
)

generated_token_ids = generated_token_ids.cpu().tolist()
print("Generation complete.")

tokens = tokenizer.decode(generated_token_ids)
tokens.dump_midi(args.output)
print(f"MIDI file saved as: {args.output}")