from PIL import Image
import io
import cairosvg

# Function to convert SVG to PNG
def convert_svg_to_png():
    try:
        # Read SVG file
        with open('assets/als_diagram.svg', 'rb') as svg_file:
            svg_data = svg_file.read()
        
        # Convert to PNG
        png_data = cairosvg.svg2png(bytestring=svg_data)
        
        # Save PNG
        with open('assets/als_diagram.png', 'wb') as png_file:
            png_file.write(png_data)
        
        print("Successfully converted SVG to PNG")
    
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        # Create a fallback image with text
        fallback_img = Image.new('RGB', (800, 450), color=(245, 245, 245))
        with open('assets/als_diagram.png', 'wb') as f:
            fallback_img.save(f, format='PNG')
        print("Created fallback image")

if __name__ == "__main__":
    convert_svg_to_png()
