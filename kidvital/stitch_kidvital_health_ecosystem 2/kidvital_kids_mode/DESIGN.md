---
name: KidVital Kids Mode
colors:
  surface: '#fbf9f8'
  surface-dim: '#dbd9d9'
  surface-bright: '#fbf9f8'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f5f3f3'
  surface-container: '#efeded'
  surface-container-high: '#eae8e7'
  surface-container-highest: '#e4e2e2'
  on-surface: '#1b1c1c'
  on-surface-variant: '#4f4635'
  inverse-surface: '#303030'
  inverse-on-surface: '#f2f0f0'
  outline: '#817663'
  outline-variant: '#d2c5af'
  surface-tint: '#795900'
  primary: '#795900'
  on-primary: '#ffffff'
  primary-container: '#ffc94d'
  on-primary-container: '#735400'
  inverse-primary: '#f4bf44'
  secondary: '#006d41'
  on-secondary: '#ffffff'
  secondary-container: '#95f7bb'
  on-secondary-container: '#007346'
  tertiary: '#006780'
  on-tertiary: '#ffffff'
  tertiary-container: '#7eddff'
  on-tertiary-container: '#006179'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#ffdf9f'
  primary-fixed-dim: '#f4bf44'
  on-primary-fixed: '#261a00'
  on-primary-fixed-variant: '#5c4300'
  secondary-fixed: '#95f7bb'
  secondary-fixed-dim: '#7adaa1'
  on-secondary-fixed: '#002110'
  on-secondary-fixed-variant: '#005230'
  tertiary-fixed: '#b7eaff'
  tertiary-fixed-dim: '#60d4fb'
  on-tertiary-fixed: '#001f28'
  on-tertiary-fixed-variant: '#004e61'
  background: '#fbf9f8'
  on-background: '#1b1c1c'
  surface-variant: '#e4e2e2'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 48px
    fontWeight: '800'
    lineHeight: 56px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '800'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
  headline-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 22px
    fontWeight: '700'
    lineHeight: 28px
  body-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 26px
  body-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '500'
    lineHeight: 24px
  label-bold:
    fontFamily: Plus Jakarta Sans
    fontSize: 14px
    fontWeight: '700'
    lineHeight: 20px
rounded:
  sm: 0.5rem
  DEFAULT: 1rem
  md: 1.5rem
  lg: 2rem
  xl: 3rem
  full: 9999px
spacing:
  unit: 8px
  gutter: 24px
  margin-mobile: 24px
  margin-desktop: 64px
  stack-sm: 12px
  stack-md: 24px
  stack-lg: 40px
---

## Brand & Style

This design system is crafted for an energetic, safe, and encouraging environment specifically for children. The brand personality is joyful and supportive, functioning as a digital playground that prioritizes ease of use and visual delight.

The design style is **Playful Flat-Cartoon**. It avoids complex gradients and realistic lighting in favor of bold, solid shapes and high-contrast outlines that mimic modern children’s illustration. Every element is oversized and tactile, evoking a sense of "squishiness" and physical safety through extreme roundedness. The goal is to make the UI feel like a set of friendly building blocks rather than a technical interface.

## Colors

The palette is vibrant and warm, designed to maintain high engagement without overstimulating. 

- **Primary (Yellow):** Used for main actions and "hero" moments. It represents energy and optimism.
- **Secondary (Green):** Used for success states, health-related progress, and "go" actions.
- **Tertiary (Blue):** Used for information, calm transitions, and secondary navigation elements.
- **Background (Pastel Cream):** A soft, eye-straining-reduction base that makes the vibrant foreground colors pop without the harshness of pure white.
- **Ink (Neutral):** A deep charcoal rather than pure black is used for text and outlines to keep the vibe soft and friendly.

## Typography

This design system utilizes **Plus Jakarta Sans** exclusively to maintain a relationship with the parent application while pivoting toward a much bolder, more "bubbly" weight distribution.

- **Weight Strategy:** Headlines should almost always use ExtraBold (800) or Bold (700). Medium (500) is the minimum weight for body text to ensure maximum legibility for developing eyes.
- **Readability:** Line heights are generous to prevent text crowding. 
- **Scale:** Type sizes are larger than standard SaaS applications to accommodate larger tap targets and lower information density.

## Layout & Spacing

The layout philosophy follows a **Fluid Content** model with extremely generous "air" between elements. 

- **Grid:** A simple 12-column grid for desktop and a 4-column grid for mobile.
- **Rhythm:** We use an 8px base unit, but the "standard" padding for cards and containers is tripled (24px) to emphasize the chunky, playful nature of the components.
- **Touch Targets:** No interactive element should be smaller than 48x48px, with a preferred size of 64px for primary navigation icons.
- **Reflow:** On mobile, all horizontal card arrays should transform into vertical stacks or side-scrolling carousels with large, visible "peek" edges.

## Elevation & Depth

This design system avoids realistic shadows and blurs. Instead, it uses **Flat Dimensionality**:

- **Tonal Stacking:** Depth is created by placing lighter-colored containers on top of slightly darker background regions.
- **Hard Shadows:** Instead of soft ambient shadows, use a "Block Shadow" (a solid offset fill, usually 4px to 8px down) in a slightly darker version of the background or a semi-transparent neutral. This gives buttons a "pressable" 3D look without using gradients.
- **Outlines:** All containers should have a subtle 2px stroke in a darker tone of the surface color to define boundaries clearly.

## Shapes

The shape language is defined by **Extreme Roundness**. There are no sharp corners in the world of this design system.

- **Base Radius:** All standard buttons and small containers use a 24px (1.5rem) radius.
- **Large Containers:** Cards and modals use a 32px to 48px radius.
- **Pills:** All labels and chips use a fully rounded (pill) shape.
- **Iconography:** Use thick-stroke (3px+) rounded icons. Avoid thin lines or sharp points.

## Components

### Buttons
Primary buttons use the "Block Shadow" technique. On hover, the button moves down 2px. On click, the button moves down 4px (the full depth of the shadow), creating a "squish" effect. Use ExtraBold type for button labels.

### Chips & Badges
Always pill-shaped with high-contrast background colors. These are used for rewards, categories, or status indicators like "Great Job!" or "Level 1".

### Cards
Cards should have a 32px corner radius and a subtle 2px border. They function as the primary containers for activities and content. Use a white or very light cream background for cards to make them pop against the #FFF8EC page background.

### Input Fields
Inputs are oversized (min-height 64px) with a 24px corner radius. The border thickens and changes color to the primary yellow when focused to provide clear visual feedback.

### Progress Bars
Extra thick (24px height) with fully rounded ends. The track should be a lighter version of the fill color, emphasizing the progress made with a bright Green (#6FCF97) fill.

### Illustrations
Incorporate "Character Spots"—small, flat-style cartoon mascots—near feedback components to provide emotional cues (e.g., a smiling sun when a task is completed).