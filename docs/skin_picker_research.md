# The skin picker: how to present a character's skins (research, 2026-10-25)

The question (user): the character SKINS menu in the Characters tab was a native `<select>` with three optgroups
(switch to a skin, "+ style" = make a new skin on the GPU, and a relabel group). The user found it unintuitive. This
note collects how other products present variants of one item that people switch between and add to, and the
principles we built the skin menu from (cinestudio/web/js/styles.js `charSkinSwitch`, `openSkinDlg`).

## What others do

- **E-commerce variant swatches (Baymard).** Baymard's testing says a native drop-down "is a very poor interface for
  visual selections"; options such as colours "are much better communicated using visual aids, like color swatches or
  color variation thumbnails" [1]. Variants of one product belong in ONE list item with swatches under it, not as
  separate items ("major usability issues with displaying variations as separate list items") [2], and the swatches
  should be shown in full on mobile too [3]. A swatch that looks available but is not frustrates people [3].
  -> One tile per character in the library (already built) and the skins as picture swatches under the name.
- **Show the options instead of hiding them (NN/g).** Listboxes "expose options right away"; drop-downs hide them
  and suit cases where alternatives should be downplayed; for fewer than ~5 options radio buttons (all visible) are
  preferred [4]. A character has 1-4 skins: all should be visible at once.
- **Game lockers and character customizers (Fortnite Locker, Valorant collection, Overwatch hero gallery).** A grid
  or strip of portrait thumbnails of the outfits of ONE character; the equipped one carries a clear mark (a frame and a
  check or "equipped" tag), hovering or selecting previews it on the character, and getting more is a separate entry
  (the shop) with its price shown before you commit [5]. (Our own observation of the products; the official docs do
  not describe the locker UI in detail.)
- **Design tools' variants (Figma).** Variants of one component live together in a component set and are switched from
  the properties panel of the instance; switching keeps the instance's overrides [6]. -> Switching a skin keeps the
  same character (name, body, kit); only the look changes, so the switch belongs right next to the name.
- **AI image tools (Midjourney Style Explorer / style references).** Styles are browsed as pictures, and "Try Style"
  applies a style to your current prompt in one action [7]; a style reference is an IMAGE, not a word list. ->
  The New skin dialog shows each library style with its reference picture (a colour swatch with the initial when it
  has none), its label and its words.
- **Long jobs (NN/g progress indicators).** Waits over ~10 s need a percent-done indicator, not a looping spinner,
  plus a rough time estimate [8]. A new skin is ~20 minutes and ~4 Qwen jobs.
- **Rare and destructive actions (NN/g).** Overflow icons (⋯, ⋮) are widely understood as "more options" and suit
  SECONDARY actions; keep them next to the item they affect and do not bury primary actions there [9]. Keep
  consequential actions away from benign ones [10], and confirm destructive actions (sparingly, so people keep reading
  the confirmations) [11].
- **Accessibility (W3C APG radio group).** A set of mutually exclusive choices is a `radiogroup` of `radio`s with
  `aria-checked`; Tab enters and leaves the group, arrow keys move between the radios (roving tabindex: only one
  radio is in the tab order) [12].

## Principles for the skin menu (as built, after the user's feedback)

The first build was a row of skin cards under the description; the user liked the feature but it pushed the sheet
down ("can we make it like a drop down that hovers over our sheets ... that way it doesnt offset our sheet"). The
built version (styles.js `charSkinSwitch`, `openSkinDlg`):

1. **No layout cost.** A small glass button (the skin's round picture, its style label, the number of skins, a caret)
   floats on the displayed picture's own top-left corner, over its black margin; the canvas does not move
   (screenshots compared pixel by pixel with and without it). On the Views grid it shrinks to picture + caret; it is
   hidden on the Mannequin view (own layout) and on the Candidates (no front yet; it would cover a head).
2. **Pictures, the current one obvious** [1][4][7]: the menu lists every skin with its portrait, style, status dot and
   "main"; the skin on show is tinted and checked (not colour alone).
3. **One choice switches** and closes the menu; the focus returns to the button.
4. **"Add" is its own row, the cost said before the commit** [5]: "+ New skin…" opens a dialog of the library styles
   (reference picture or a colour square with the initial, label, words). Short copy (user: "way too verbose ... others
   may be using different gpus"): "Redraws this character in another style. The pose stays." and "About 4 image jobs ·
   added to the queue (N ahead)"; no minutes, no GPU names. "Check its sheet, then Use as is" comes as the toast at the end.
5. **A long job shows in place** [8]: a skin being made is a row with "waiting · N ahead" / "picture N of ~4" and a bar.
6. **Rare actions at the bottom of the menu** [9][10][11]: "Delete this skin…" (confirmed) and, only for a skin without
   a library style, "Set which style this skin is drawn in…" (words only).
7. **Scale and long text:** the menu's list scrolls (New skin and the actions stay pinned), opens with the current skin
   in view, stays inside the window (shift/flip); the dialog has a fixed max height with its own scroll area, a filter
   field from 9 styles, free styles first and "Already skins of this character" after, 1-3 columns so 1-3 styles look
   intentional, an empty state with "Make a style"; labels clamp to 1-2 lines with the full text in a tooltip.
8. **Keyboard** [12]: menu button pattern (aria-haspopup/expanded, menuitemradio for skins; ArrowDown/Up/Enter/Space
   open, arrows/Home/End move, Esc closes and returns the focus, Tab leaves). Dialog: a radiogroup where arrows move
   and choose (by grid rows), Enter or Space chooses; only the Make button starts the GPU job.

Display framing (user: "the extra gap feels off"): the Sheet view shows the sheet cropped to its content box (non-black
pixels, an even margin, extra room at the top for the button) with CSS object-view-box; the sheet FILE keeps its
one-scale margins (AGENTS.md 4.5). The Front and Views are not cropped (the region tool's box uses the full image).

## Not decided / open

- No hover preview of a skin on the big canvas (the lockers do it).
- Not tested with real users beyond the user's own feedback rounds.

## Sources

1. Baymard Institute, "5 Common Usability Pitfalls of Custom Designed Drop-Downs" — https://baymard.com/blog/custom-dropdowns-cause-issues
2. Baymard Institute, "Combine Variations of Products into One List Item" — https://baymard.com/blog/combine-variations-one-list-item
3. Baymard Institute, "Make All Color Swatches Available in Mobile List Items for Visually Driven Product Types" — https://baymard.com/blog/mobile-interactive-color-swatches
4. Nielsen Norman Group, "Listboxes vs. Dropdown Lists" — https://www.nngroup.com/articles/listbox-dropdown/
5. Fortnite Locker / Valorant collection / Overwatch hero gallery: product observation (no official UI documentation found).
6. Figma Help Center, "Create and use variants" / instance swap — https://help.figma.com/hc/en-us/articles/360039150733
7. Midjourney docs, "Style Reference" and Style Explorer — https://docs.midjourney.com/hc/en-us/articles/32180011136653-Style-Reference
8. Nielsen Norman Group, "Progress Indicators Make a Slow System Less Insufferable" — https://www.nngroup.com/articles/progress-indicators/
9. Nielsen Norman Group, "Designing Effective Contextual Menus: 10 Guidelines" — https://www.nngroup.com/articles/contextual-menus-guidelines/ and "Overflow Menu Icons" (video) — https://www.nngroup.com/videos/overflow-menu-icons/
10. Nielsen Norman Group, "Dangerous UX: Consequential Options Close to Benign Options" — https://www.nngroup.com/articles/proximity-consequential-options/
11. Nielsen Norman Group, "Preventing User Errors: Avoiding Conscious Mistakes" — https://www.nngroup.com/articles/user-mistakes/
12. W3C WAI-ARIA Authoring Practices, "Radio Group Pattern" — https://www.w3.org/WAI/ARIA/apg/patterns/radio/
