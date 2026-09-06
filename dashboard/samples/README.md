# Sample images for the hosted demo

Drop 3–5 JPEGs here and the demo offers them in a dropdown, so a visitor sees
the system working before they upload anything of their own. An empty demo
shows nothing.

Good choices, because they exercise different branches:

| File | What it demonstrates |
|------|----------------------|
| `bottle.jpg` | ambiguity the system resolves — PET or glass, same bin |
| `cup.jpg` | ambiguity it refuses — plastic, paper or ceramic span two bins |
| `banana.jpg` | a clean high-confidence sort |
| `phone.jpg` | e-waste, kept out of both dry and wet streams |
| `battery.jpg` | the honest failure: not a class the model knows |

One object per photo, plain background, good light. The filename becomes the
label in the dropdown, so `wine_glass.jpg` reads as "wine glass".

These are committed, so do not put anything here you would not publish —
a photograph of your desk with people in it is exactly what not to upload.
