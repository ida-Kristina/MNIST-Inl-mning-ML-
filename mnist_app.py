import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib

from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Draw", page_icon="🔢", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("mnist_best_model.joblib")

def canvas_to_model_input(canvas_rgba: np.ndarray):
    """
    RGBA canvas -> MNIST-lik input (1,784) + preview (28,28)
    Steg:
    1) RGBA -> gråskala på vit bakgrund
    2) Invertera till "vit siffra på svart"
    3) Crop runt siffran
    4) Skala till ~20 px
    5) Pad till 28x28
    6) Flytta till mitten
    7) Gör om bilden till samma format som orginaldatat
    """
    # RGBA -> L på vit bakgrund
    img = Image.fromarray(canvas_rgba.astype(np.uint8), mode="RGBA")
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white_bg, img).convert("L")

    arr = np.array(img).astype(np.float32)

    # Invertera (canvas: svart på vitt, MNIST: vitt på svart)
    arr = 255.0 - arr

    # Normalisera på samma sätt som jag gjorde i modellen
    arr01 = arr / 255.0

    # Cropa bort allt utom där det är ritat, dvs siffran
        # Tröskel - vad som räknas som "bläck"
    thr = 0.2

        # Massa olika sätt att kolla om man har ritat tillräckligt mycket på canvas för att det ska räknas som en siffra 
        # Har laborerat med lite olika värden och bedömer att det här blev bra balans  mellan att fortfarande få siffor vid små siffror och små penselstorlekar 
        # utan att alltid precicera en 4 när man ritar en prick. (vilket den fortfarande gör på stora penselstorlekar 🫠)
    ink = arr01 > thr
    ink_count = int(ink.sum())

    if ink_count < 200:   
        return None, None, "För lite ritat – gör strecken tjockare eller rita större."
        
    
    ys, xs = np.where(ink)
    
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_area = bbox_w * bbox_h

    if bbox_area < 600:
        return None, None, "För liten markering – rita en hel siffra."
    
    # ink_mass = float(arr01[ink].sum())
    # if ink_mass < 8.0:
    #     return None, None, "För svagt/otydligt – rita mörkare/tjockare."


    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # Lägg lite padding runt siffran, för säkerhets skull
    pad = 10
    x0 = max(x0 - pad, 0)
    y0 = max(y0 - pad, 0)
    x1 = min(x1 + pad, arr01.shape[1] - 1)
    y1 = min(y1 + pad, arr01.shape[0] - 1)

    cropped = arr01[y0:y1+1, x0:x1+1]  

    # Kolla vilken sida är längst och sett den till 20 pixlar (för att behålla rätt proportioner), som i MNIST. 
    # Sätt andra sidan till storlek som behåller proportionerna i orginalbilden 
    h, w = cropped.shape
    target = 20
    if h > w:
        new_h = target
        new_w = max(1, int(round(w * (target / h))))
    else:
        new_w = target
        new_h = max(1, int(round(h * (target / w))))

    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8), mode="L")               # Gör en bild från arrayen för skalning
    resized_img = cropped_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)     # Skalar bilden med Pillow enligt nya h, w
    resized = np.array(resized_img).astype(np.float32) / 255.0                              # Gör array utifrån skalade bilden, som kan skickas vidare till pipelinen

    # Pad-a canvaset till 28x28 med orginalinputtet i mitten
    canvas28 = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas28[top:top+new_h, left:left+new_w] = resized

    # Flytta siffran till mitten, som i MNIST
    yy, xx = np.indices((28, 28))   # ger x och y kordinater till alla pixlar
    mass = canvas28.sum()           # ljus pixel --> hög massa 

    if mass > 0:                    
        cy = (yy * canvas28).sum() / mass       # räknar ut genomsnittlig y-position för bläcket
        cx = (xx * canvas28).sum() / mass       # räknar ut genomsnittlig x-position för bläcket

        # räknar ut hur långt ifrån mitten bläcket är, baserat på mittpunkten
        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))

        # gör ett nytt canvas att fylla 
        shifted = np.zeros_like(canvas28)

        # Kollar vilka rader och kolumnen som ska kopieras över - med min och max för att vi inte ska plocka något utanför canvaset
        # Om vi ska flytta bilden nedåt (shift_y), så riskerar de nedersta raderna att hamna utanför. Då måste vi sluta tidigare när vi kopierar från originalet. 
        # Om vi ska flytta uppåt (negativ shift_y), så riskerar de översta raderna att hamna utanför, och då måste vi börja längre ner i originalet.
        src_y0 = max(0, -shift_y)
        src_y1 = min(28, 28 - shift_y)
        src_x0 = max(0, -shift_x)
        src_x1 = min(28, 28 - shift_x)

        # vart i bilden ska vi klistra in våra kopierade pixlar 
        dst_y0 = max(0, shift_y)
        dst_y1 = min(28, 28 + shift_y)
        dst_x0 = max(0, shift_x)
        dst_x1 = min(28, 28 + shift_x)

        # använder slicing för att plocka ut och klistra in de valda bitarna 
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = canvas28[src_y0:src_y1, src_x0:src_x1]
        canvas28 = shifted

    # Gör om bilden till en rad med 784 värden, för att matcha orginaldatan (med -1 så att den själv räknar ut hur många kolumner som behövs)
    x = canvas28.reshape(1, -1).astype(np.float32)
    return x, canvas28, None


st.title("Rita en siffra")
st.write("Rita en tydlig siffra (0–9). Klicka sedan på **Prediktera**.")

model = load_model()

# UI-kontroller
col1, col2 = st.columns(2)
with col1:
    brush_size = st.slider("Penselstorlek", 8, 40, 18)

# Canvas
canvas_size = 400

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",   # transparent “fyll”
    stroke_width=brush_size,
    stroke_color="#000000",          # ritar svart
    background_color="#FFFFFF",      # vit bakgrund
    width=canvas_size,
    height=canvas_size,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediktera
if st.button("Prediktera"):
    if canvas_result.image_data is None:    # Felmeddelande ifall Streamlit inte skickar någon bild
        st.warning("Ingen bild från canvasen.")
    else:
        x, preview_28, msg = canvas_to_model_input(canvas_result.image_data)

        if x is None:  # Check för att canvasen inte ska vara tom. 
            st.warning(msg if msg else "Rita en siffra först")
        else:
            pred = model.predict(x)[0]

            st.subheader("Resultat")
            st.metric("Predikterad siffra", int(pred))

            st.subheader("Det modellen ser (28×28)")
            st.image(preview_28, clamp=True, use_container_width=False)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)[0]
                st.write("Sannolikheter per klass (0–9):")
                st.bar_chart(proba)
            else:
                st.info("Ingen predict_proba")
