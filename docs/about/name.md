# The name

```{image} ../assets/elome_tengwar_light.svg
:alt: Elómë in Tengwar
:class: only-light
:height: 64px
```

```{image} ../assets/elome_tengwar_dark.svg
:alt: Elómë in Tengwar
:class: only-dark
:height: 64px
```

*Elómë* is Quenya, the High-Elven tongue of Tolkien's legendarium:
*él*, "star", and *lómë*, "dusk, dimness". It means a star dimmed, which
is what a transit is. Tolkien derived *él* (and *elen*, as in
*Elen síla*) from *ele*, "behold!", the Elves' first cry on seeing the
stars. *Lómë* is the dusk of *Lómion*, "son of the twilight".

The package is spelled `elome` in code.

## Pronunciation

/ɛˈloː.mɛ/, "eh-**LOH**-meh", in three syllables, *e-ló-më*.

- **The stress falls on *ló*.** Quenya stresses the second-to-last
  syllable when its vowel is long, and the accent on *ó* marks it as long.
  Tolkien's accents mark length, not stress.
- **The *ó* is a pure, held "o"**, as in Italian or Spanish, not the
  English glide "oh-w".
- **The final *ë* is spoken**, as a short "eh". The diaeresis only says that
  it is not silent, as in *Manwë*.
- **Both *e*s are short, open "e"s**, as in "bet".

## In Tengwar

The mark above is *Elómë* in Tengwar, Tolkien's Elvish script, in the
classical Quenya mode. The vowels are *tehtar* written above the
preceding consonant, and a vowel with no consonant before it, or a long
one, sits on a carrier:

| sound | Tengwar |
|---|---|
| E | short carrier + acute (the *e* tehta) |
| l | *lambe* |
| ó | long carrier + right curl (long *o*) |
| më | *malta* + acute |

The images are glyph outlines, made by `docs/assets/render_tengwar.py`.
HarfBuzz shapes the string, placing each tehta over its letter, and
fontTools draws the glyphs. The font is
[Alcarin Tengwar](https://github.com/Tosche/Alcarin-Tengwar) by Toshi
Omagari, under the SIL Open Font License 1.1. Tengwar is not part of
Unicode proper, which is why the name is shown as an image.
