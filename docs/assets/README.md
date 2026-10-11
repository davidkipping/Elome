# The name in Tengwar

`elome_tengwar_light.svg` and `elome_tengwar_dark.svg` spell **Elómë** in
Tengwar, Tolkien's Elvish script, in the classical Quenya mode: the vowels
are *tehtar* written above the preceding consonant, and a vowel with no
consonant before it, or a long one, sits on a carrier.

| sound | Tengwar |
|---|---|
| E | short carrier + acute (the *e* tehta) |
| l | *lambe* |
| ó | long carrier + right curl (long *o*) |
| më | *malta* + acute |

The SVGs are glyph outlines, so no font is needed to view them. They were
made by `render_tengwar.py`: HarfBuzz shapes the string (placing each
tehta over its letter) and fontTools draws the positioned glyphs. The font
is [Alcarin Tengwar](https://github.com/Tosche/Alcarin-Tengwar) by Toshi
Omagari, under the SIL Open Font License 1.1, in the Free Tengwar Font
Project's private-use encoding. Tengwar is not in Unicode proper, which is
why the title shows an image rather than text.

```bash
pip install uharfbuzz fonttools
# with AlcarinTengwar-Regular.ttf beside the script:
python render_tengwar.py
```
