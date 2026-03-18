# GE Dryer Front Bearing (WE3M26) - Fusion 360 Script
# MVP VERSION
#
# Structure:
# - Solid curved body (cradle for drum)
# - Flat bottom sits on metal frame
# - Lamp POCKET from bottom (for lamp housing)
# - Small WINDOW through curved surface (for light)
# - Curved drum surface stays intact except small window

import adsk.core, adsk.fusion, traceback
import math

# =============================================================================
# DIMENSIONS IN MM
# =============================================================================

# Main body
BODY_LENGTH_MM = 287.0          # Length (X direction)
BODY_DEPTH_MM = 49.0            # Front-to-back depth (Y direction)
BODY_HEIGHT_MM = 35.0           # Height to top of curve (Z)

# Drum curve
DRUM_RADIUS_MM = 304.8          # 12 inches

# Lamp pocket (recessed area from bottom for lamp housing)
LAMP_POCKET_WIDTH_MM = 68.0     # Width (X)
LAMP_POCKET_LENGTH_MM = 46.0    # Length (Y)
LAMP_POCKET_DEPTH_MM = 15.0     # How deep from bottom (leaves ~20mm at top)
LAMP_OFFSET_X_MM = 20.0         # Offset from center toward right

# Light window (small hole through curved surface for light)
LIGHT_WINDOW_SIZE_MM = 25.0     # Size of window (square)

# =============================================================================
def mm(val):
    return val / 10.0


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox('No active Fusion design', 'Error')
            return

        rootComp = design.rootComponent

        # Create component
        occ = rootComp.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        comp = occ.component
        comp.name = "Dryer_Bearing_MVP"

        # Step 1: Create solid curved body
        body = create_solid_body(comp)

        if body:
            # Step 2: Cut lamp pocket (from bottom, partial depth)
            cut_lamp_pocket(comp, body)

            # NOTE: Removed light window cut - it was removing the curved surface
            # If needed, add a small hole later

            ui.messageBox(
                f'Bearing Created!\n\n'
                f'Size: {BODY_LENGTH_MM:.0f} × {BODY_DEPTH_MM:.0f} × {BODY_HEIGHT_MM:.0f}mm\n'
                f'Lamp pocket: {LAMP_POCKET_WIDTH_MM:.0f} × {LAMP_POCKET_LENGTH_MM:.0f}mm\n'
                f'Light window: {LIGHT_WINDOW_SIZE_MM:.0f}mm',
                'Success'
            )

    except:
        if ui:
            ui.messageBox(f'Failed:\n{traceback.format_exc()}', 'Error')


def create_solid_body(comp):
    """
    Create solid curved body.
    Simple profile: flat bottom, curved top, straight sides.
    """
    sketches = comp.sketches

    sketch = sketches.add(comp.xZConstructionPlane)
    sketch.name = "Body_Profile"

    lines = sketch.sketchCurves.sketchLines
    arcs = sketch.sketchCurves.sketchArcs

    # Dimensions in cm
    half_length = mm(BODY_LENGTH_MM) / 2
    height = mm(BODY_HEIGHT_MM)
    drum_r = mm(DRUM_RADIUS_MM)

    # Calculate arc geometry
    # Arc center is at (0, 0, height - drum_r)
    # At x = half_length, z = center_z + sqrt(R² - x²)
    arc_center_z = height - drum_r

    if drum_r > half_length:
        edge_z = arc_center_z + math.sqrt(drum_r * drum_r - half_length * half_length)
    else:
        edge_z = 0
    edge_z = max(0, edge_z)

    # Profile points
    bottom_left = adsk.core.Point3D.create(-half_length, 0, 0)
    bottom_right = adsk.core.Point3D.create(half_length, 0, 0)
    top_left = adsk.core.Point3D.create(-half_length, 0, edge_z)
    top_right = adsk.core.Point3D.create(half_length, 0, edge_z)
    top_center = adsk.core.Point3D.create(0, 0, height)

    # Draw closed profile
    lines.addByTwoPoints(bottom_left, bottom_right)   # Bottom
    lines.addByTwoPoints(bottom_right, top_right)     # Right side
    arcs.addByThreePoints(top_right, top_center, top_left)  # Curved top
    lines.addByTwoPoints(top_left, bottom_left)       # Left side

    if sketch.profiles.count == 0:
        raise Exception("Body_Profile: No closed profile")

    profile = sketch.profiles.item(0)

    # Extrude
    extrudes = comp.features.extrudeFeatures
    depth = mm(BODY_DEPTH_MM)

    extInput = extrudes.createInput(profile, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    extInput.setDistanceExtent(False, adsk.core.ValueInput.createByReal(depth))

    extrude = extrudes.add(extInput)
    extrude.name = "Body_Extrude"

    body = extrude.bodies.item(0)
    body.name = "Bearing_Body"

    return body


def cut_lamp_pocket(comp, body):
    """
    Cut lamp pocket from BOTTOM going UP.
    Creates a pocket in the flat bottom face, leaving curved top intact.
    """
    try:
        sketches = comp.sketches
        extrudes = comp.features.extrudeFeatures

        # Dimensions
        pocket_w = mm(LAMP_POCKET_WIDTH_MM)
        pocket_l = mm(LAMP_POCKET_LENGTH_MM)
        pocket_d = mm(LAMP_POCKET_DEPTH_MM)
        offset_x = mm(LAMP_OFFSET_X_MM)
        body_depth = mm(BODY_DEPTH_MM)

        # Create sketch on an OFFSET plane at Z = pocket_depth
        # Then cut DOWNWARD to Z=0 (this ensures correct direction)
        planes = comp.constructionPlanes
        planeInput = planes.createInput()
        planeInput.setByOffset(
            comp.xYConstructionPlane,
            adsk.core.ValueInput.createByReal(pocket_d)
        )
        pocketPlane = planes.add(planeInput)
        pocketPlane.name = "Lamp_Pocket_Plane"

        sketch = sketches.add(pocketPlane)
        sketch.name = "Lamp_Pocket_Profile"

        lines = sketch.sketchCurves.sketchLines

        # Rectangle centered at offset position
        x_left = offset_x - pocket_w / 2
        x_right = offset_x + pocket_w / 2
        y_front = (body_depth - pocket_l) / 2
        y_back = (body_depth + pocket_l) / 2

        p1 = adsk.core.Point3D.create(x_left, y_front, 0)
        p2 = adsk.core.Point3D.create(x_right, y_back, 0)
        lines.addTwoPointRectangle(p1, p2)

        if sketch.profiles.count == 0:
            print("Lamp_Pocket_Profile: No profile")
            return

        profile = sketch.profiles.item(0)

        # Cut DOWNWARD from the offset plane to bottom
        cutInput = extrudes.createInput(profile, adsk.fusion.FeatureOperations.CutFeatureOperation)
        # One-sided extent, negative direction (downward)
        extent = adsk.fusion.DistanceExtentDefinition.create(adsk.core.ValueInput.createByReal(pocket_d))
        cutInput.setOneSideExtent(extent, adsk.fusion.ExtentDirections.NegativeExtentDirection)
        cutInput.participantBodies = [body]

        cut = extrudes.add(cutInput)
        cut.name = "Lamp_Pocket_Cut"

    except Exception as e:
        print(f"Lamp pocket error: {e}")


def cut_light_window(comp, body):
    """
    Cut small window through curved surface for light.
    This is a small hole that goes all the way through.
    """
    try:
        sketches = comp.sketches
        extrudes = comp.features.extrudeFeatures

        # Sketch on XY plane at Z=0
        sketch = sketches.add(comp.xYConstructionPlane)
        sketch.name = "Light_Window_Profile"

        lines = sketch.sketchCurves.sketchLines

        # Small square window centered in lamp pocket area
        window_size = mm(LIGHT_WINDOW_SIZE_MM)
        offset_x = mm(LAMP_OFFSET_X_MM)
        body_depth = mm(BODY_DEPTH_MM)

        half_w = window_size / 2
        center_y = body_depth / 2

        p1 = adsk.core.Point3D.create(offset_x - half_w, center_y - half_w, 0)
        p2 = adsk.core.Point3D.create(offset_x + half_w, center_y + half_w, 0)
        lines.addTwoPointRectangle(p1, p2)

        if sketch.profiles.count == 0:
            print("Light_Window_Profile: No profile")
            return

        profile = sketch.profiles.item(0)

        # Cut ALL the way through
        cutInput = extrudes.createInput(profile, adsk.fusion.FeatureOperations.CutFeatureOperation)
        cutInput.setAllExtent(adsk.fusion.ExtentDirections.PositiveExtentDirection)
        cutInput.participantBodies = [body]

        cut = extrudes.add(cutInput)
        cut.name = "Light_Window_Cut"

    except Exception as e:
        print(f"Light window error: {e}")


def main():
    run(None)

if __name__ == '__main__':
    main()
