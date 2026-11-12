import ollama
from tabulate import tabulate

def test_model_temperatures(prompt: str, model_name: str = "qwen"):
    # Define different temperature levels to test
    temperatures = [0.0, 0.5, 1.0, 1.5]
    
    # Dictionary describing what to expect at a given temperature
    descriptions = {
        0.0: "deterministic",
        0.5: "balanced",
        1.0: "creative",
        1.5: "crazy"
    }
    
    table_data = []
    
    print(f"Sending prompt to model '{model_name}'...")
    print(f"Prompt: \"{prompt}\"\n")
    print(f"Generating responses for different temperatures...")
    
    for temp in temperatures:
        try:
            # Call Ollama with a specific temperature
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temp,
                    "num_predict": 150 # Limit length to keep the table readable
                }
            )
            
            output_text = response["message"]["content"].strip()
            
            # Add header and description to the result so it looks nice in the table
            header = f"TEMP: {temp}\n({descriptions[temp]})"
            table_data.append([header, output_text])
            
        except Exception as e:
            table_data.append([f"TEMP: {temp}", f"Error: {str(e)}"])

    # Print results as a readable table with grid lines
    print("\n" + "="*80)
    print(" TEMPERATURE COMPARISON RESULTS")
    print("="*80)
    print(tabulate(table_data, headers=["Setting", "Model Response"], tablefmt="grid"))

if __name__ == "__main__":
    test_prompt = "Write a short paragraph on the Solar System."
    test_model_temperatures(prompt=test_prompt, model_name="qwen")

