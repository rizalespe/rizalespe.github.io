---
name: KidVital Health
colors:
  surface: '#fcf9f8'
  surface-dim: '#dcd9d9'
  surface-bright: '#fcf9f8'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f6f3f2'
  surface-container: '#f0eded'
  surface-container-high: '#eae7e7'
  surface-container-highest: '#e4e2e1'
  on-surface: '#1b1c1c'
  on-surface-variant: '#424751'
  inverse-surface: '#303030'
  inverse-on-surface: '#f3f0f0'
  outline: '#727782'
  outline-variant: '#c2c6d3'
  surface-tint: '#155fa9'
  primary: '#105da7'
  on-primary: '#ffffff'
  primary-container: '#3776c1'
  on-primary-container: '#fefcff'
  inverse-primary: '#a5c8ff'
  secondary: '#575f68'
  on-secondary: '#ffffff'
  secondary-container: '#dbe3ed'
  on-secondary-container: '#5d656e'
  tertiary: '#765700'
  on-tertiary: '#ffffff'
  tertiary-container: '#956e00'
  on-tertiary-container: '#fffbff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d4e3ff'
  primary-fixed-dim: '#a5c8ff'
  on-primary-fixed: '#001c3a'
  on-primary-fixed-variant: '#004785'
  secondary-fixed: '#dbe3ed'
  secondary-fixed-dim: '#bfc7d1'
  on-secondary-fixed: '#141c23'
  on-secondary-fixed-variant: '#404850'
  tertiary-fixed: '#ffdf9f'
  tertiary-fixed-dim: '#f4bf44'
  on-tertiary-fixed: '#261a00'
  on-tertiary-fixed-variant: '#5c4300'
  background: '#fcf9f8'
  on-background: '#1b1c1c'
  surface-variant: '#e4e2e1'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 40px
    fontWeight: '700'
    lineHeight: 48px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Work Sans
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Work Sans
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-md:
    fontFamily: Work Sans
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
  kids-display:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '800'
    lineHeight: 40px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 8px
  container-margin: 24px
  gutter: 16px
  section-gap: 32px
  stack-sm: 8px
  stack-md: 16px
  stack-lg: 24px
---

## Brand & Style
The design system bridges the gap between clinical precision and domestic warmth. It serves two distinct personas: the **Parent/Caregiver**, who requires clarity, trust, and rapid access to medical data; and the **Child**, who needs an engaging, low-anxiety environment to track wellness habits.

The aesthetic is **Modern Minimalist with a Tactile Softness**. It avoids the sterility of traditional medical apps by using organic curves and a warm base palette. In Parent Mode, the UI is professional and structured. In Kids Mode, the system shifts toward a playful, "toy-like" interface using high-contrast primary colors and exaggerated geometry to encourage interaction.

**Key Visual Pillars:**
- **Clarity over Clutter:** Essential health metrics are prioritized with generous whitespace.
- **Approachable Authority:** Professional blue tones paired with soft, pill-shaped elements.
- **Safety First:** High-contrast alerts and large tap targets (min 48px) for high-stress medical situations.

## Colors
The palette is divided into two functional modes to signal the current context of the application.

### Parent & Medical Mode
- **Primary Action (#3D7BC7):** "Trusted Blue" for all primary buttons, active states, and navigation cues.
- **Surface (#E8F0FA):** A soft blue tint used for card backgrounds and secondary containers to group medical data.
- **Neutral (#F7F3EC):** A warm off-white background that reduces eye strain compared to pure white.
- **Status:** Clear differentiation between Emergency (#E4572E) and Success (#2E9E6D).

### Kids Mode
The palette shifts to a vibrant, triadic scheme. 
- **The Core Trio:** Yellow (#FFC94D), Green (#6FCF97), and Sky Blue (#56CCF2) are used as primary fills for interactive elements and character-based rewards.
- **Background:** Softens to a pastel cream (#FFF8EC) to maintain warmth.

### Dark Mode (Parent Only)
The background shifts to #1C1C1E. Elevation is handled through subtle opacity layers rather than pure black to maintain a premium, high-contrast feel that meets WCAG AA standards.

## Typography
The system uses a duo-font strategy to balance personality and utility.

**Plus Jakarta Sans** is used for headlines. Its soft, rounded terminals feel friendly and modern, perfectly matching the "warmth" of the brand. In Kids Mode, weight is pushed to "ExtraBold" (800) to create a more impactful, playful feel.

**Work Sans** is used for all body text, data points, and labels. It is chosen for its exceptional legibility at small sizes, which is critical for reading medical dosages or history logs.

- **Scale:** On mobile, headlines should scale down by 20%, but body sizes remain at 16px/18px to ensure accessibility for tired parents or elderly caregivers.
- **Letter Spacing:** Headlines use slight negative tracking (-0.02em) to appear more cohesive.

## Layout & Spacing
The layout follows a **Fluid Grid** model based on an 8px base unit. 

- **Parent Mode:** Uses a 24px side margin to create a "contained" and secure feel. Content is grouped into cards with 16px internal padding.
- **Kids Mode:** Spacing increases to a 32px vertical rhythm to prevent accidental taps and create a "roomy," low-stress environment.

**Adaptive Behavior:**
- **Mobile:** Single column focus. Bottom navigation is persistent.
- **Tablet:** 2-column layout. The left column handles navigation and "Vito" (the mascot) in Kids Mode, while the right handles active data entry or tracking.

## Elevation & Depth
Depth is used to signify "interactivity" and "urgency."

- **Parent Mode (Tonal Layers):** Uses a very soft, diffused shadow (15% opacity, 20px blur) to lift cards off the neutral background. Interactive elements have a subtle "pressed" state to provide haptic-visual feedback.
- **Kids Mode (Flat/High Contrast):** Elevation is conveyed via **High-Contrast Outlines** and bold color blocks rather than shadows. This creates a "cartoon" aesthetic that feels approachable and tactile without the complexity of realistic depth.
- **Alerts:** Emergency alerts (e.g., missed medication) use a high-contrast shadow with a subtle color tint (#E4572E) to draw immediate attention.

## Shapes
Shapes are fundamentally rounded to evoke safety and kindness.

- **Parent Mode:** Standard buttons and cards use a 16px radius (rounded-lg). This provides a modern, friendly feel while maintaining enough structure for dense medical information.
- **Kids Mode:** The radius increases to 28px+ or "Full Pill" for all interactive elements. This exaggerated roundness mimics the physical design of children's toys and removes any "sharp" or intimidating edges.
- **Icons:** All iconography must use rounded caps and corners. Sharp 90-degree angles are prohibited.

## Components

### Buttons
- **Parent Action:** Blue fill, white text, 16px radius. High-contrast, no-nonsense.
- **Kids Action:** Bright Yellow or Green fill, thick 2pt stroke (Neutral #2B2B2B), 32px radius. 
- **Tap Targets:** Every button has a minimum height of 48px in Parent Mode and 56px in Kids Mode.

### Cards
- **Medical Log:** White background, subtle shadow, 16px radius. Left-aligned typography.
- **Activity Card (Kids):** Sky Blue background, bold outline, no shadow. Center-aligned typography with "Vito" mascot icons.

### Inputs & Forms
- Inputs use a "Floating Label" pattern to save space while ensuring the context is never lost.
- Checkboxes and Radio buttons are oversized (24px) for easier interaction.

### Navigation
- **Bottom Bar:** Fixed at the bottom with 4 primary icons. In Kids Mode, icons are larger and accompanied by colorful circular backgrounds when active.

### Mascot Integration
- **Vito the Fox:** Vito appears as a celebratory feedback component (e.g., "Great job drinking water!") and as a guide in onboarding. His visual style is flat, vector-based, and uses the primary kids' palette.