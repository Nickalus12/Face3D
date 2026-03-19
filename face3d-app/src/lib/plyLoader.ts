/**
 * PLY binary_little_endian parser for Gaussian Splat outputs.
 *
 * Handles the format exported by gsplat:
 *   - Positions: x, y, z (float32)
 *   - SH DC colors: f_dc_0, f_dc_1, f_dc_2 (float32)
 *   - Opacity: opacity (float32)
 *
 * SH DC -> RGB conversion:
 *   C0 = 0.28209479177387814  (1 / (2 * sqrt(pi)))
 *   rgb = sigmoid(sh_dc * C0 + 0.5)
 */

export interface PlyData {
  positions: Float32Array;
  colors: Float32Array;
  opacities: Float32Array | null;
  count: number;
  fileSize: number;
}

interface PlyProperty {
  name: string;
  type: string;
  byteSize: number;
}

const TYPE_SIZES: Record<string, number> = {
  char: 1,
  uchar: 1,
  short: 2,
  ushort: 2,
  int: 4,
  uint: 4,
  float: 4,
  double: 8,
  int8: 1,
  uint8: 1,
  int16: 2,
  uint16: 2,
  int32: 4,
  uint32: 4,
  float32: 4,
  float64: 8,
};

function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x));
}

/**
 * Parse PLY header from raw bytes.
 * Returns header info and the byte offset where binary data starts.
 */
function parseHeader(buffer: ArrayBuffer): {
  vertexCount: number;
  properties: PlyProperty[];
  dataOffset: number;
  format: string;
} {
  // Read the header as text (headers are always ASCII)
  const headerBytes = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 8192));
  const headerText = new TextDecoder("ascii").decode(headerBytes);

  const endHeaderIdx = headerText.indexOf("end_header\n");
  if (endHeaderIdx === -1) {
    // Try \r\n
    const endHeaderIdx2 = headerText.indexOf("end_header\r\n");
    if (endHeaderIdx2 === -1) {
      throw new Error("Invalid PLY file: no end_header found");
    }
    return parseHeaderFromText(
      headerText.substring(0, endHeaderIdx2),
      endHeaderIdx2 + "end_header\r\n".length,
    );
  }

  return parseHeaderFromText(
    headerText.substring(0, endHeaderIdx),
    endHeaderIdx + "end_header\n".length,
  );
}

function parseHeaderFromText(
  headerText: string,
  dataOffset: number,
): {
  vertexCount: number;
  properties: PlyProperty[];
  dataOffset: number;
  format: string;
} {
  const lines = headerText.split(/\r?\n/);
  let vertexCount = 0;
  let format = "";
  const properties: PlyProperty[] = [];
  let inVertexElement = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("format ")) {
      format = trimmed.split(/\s+/)[1];
    } else if (trimmed.startsWith("element vertex")) {
      vertexCount = parseInt(trimmed.split(/\s+/)[2], 10);
      inVertexElement = true;
    } else if (trimmed.startsWith("element ")) {
      inVertexElement = false;
    } else if (trimmed.startsWith("property ") && inVertexElement) {
      const parts = trimmed.split(/\s+/);
      // property <type> <name>
      if (parts.length >= 3 && parts[1] !== "list") {
        const typeName = parts[1];
        const propName = parts[2];
        const byteSize = TYPE_SIZES[typeName];
        if (byteSize === undefined) {
          throw new Error(`Unknown PLY type: ${typeName}`);
        }
        properties.push({ name: propName, type: typeName, byteSize });
      }
    }
  }

  if (vertexCount === 0) {
    throw new Error("PLY file has no vertices");
  }

  return { vertexCount, properties, dataOffset, format };
}

/**
 * Parse a binary_little_endian PLY file into positions, colors, and opacities.
 * Handles files with 500K+ points by processing in a tight loop.
 */
export function parsePlyBuffer(buffer: ArrayBuffer): PlyData {
  const { vertexCount, properties, dataOffset, format } = parseHeader(buffer);

  if (format !== "binary_little_endian") {
    throw new Error(
      `Unsupported PLY format: "${format}". Only binary_little_endian is supported.`,
    );
  }

  // Calculate vertex stride and find property offsets
  let stride = 0;
  const propOffsets: Record<string, { offset: number; type: string }> = {};

  for (const prop of properties) {
    propOffsets[prop.name] = { offset: stride, type: prop.type };
    stride += prop.byteSize;
  }

  // Verify required properties exist
  const requiredPos = ["x", "y", "z"];
  for (const name of requiredPos) {
    if (!(name in propOffsets)) {
      throw new Error(`PLY file missing required property: ${name}`);
    }
  }

  // Check for SH DC color properties (gsplat format)
  const hasSHColor =
    "f_dc_0" in propOffsets &&
    "f_dc_1" in propOffsets &&
    "f_dc_2" in propOffsets;

  // Check for direct RGB colors (0-255 uchar)
  const hasRGBColor =
    "red" in propOffsets && "green" in propOffsets && "blue" in propOffsets;

  const hasOpacity = "opacity" in propOffsets;

  // Allocate output arrays
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const opacities = hasOpacity ? new Float32Array(vertexCount) : null;

  // SH coefficient C0 = 1 / (2 * sqrt(pi))
  const C0 = 0.28209479177387814;

  const dataView = new DataView(buffer, dataOffset);

  // Parse vertices in a tight loop
  for (let i = 0; i < vertexCount; i++) {
    const base = i * stride;

    // Positions
    positions[i * 3] = dataView.getFloat32(
      base + propOffsets["x"].offset,
      true,
    );
    positions[i * 3 + 1] = dataView.getFloat32(
      base + propOffsets["y"].offset,
      true,
    );
    positions[i * 3 + 2] = dataView.getFloat32(
      base + propOffsets["z"].offset,
      true,
    );

    // Colors
    if (hasSHColor) {
      const sh0 = dataView.getFloat32(
        base + propOffsets["f_dc_0"].offset,
        true,
      );
      const sh1 = dataView.getFloat32(
        base + propOffsets["f_dc_1"].offset,
        true,
      );
      const sh2 = dataView.getFloat32(
        base + propOffsets["f_dc_2"].offset,
        true,
      );
      colors[i * 3] = sigmoid(sh0 * C0 + 0.5);
      colors[i * 3 + 1] = sigmoid(sh1 * C0 + 0.5);
      colors[i * 3 + 2] = sigmoid(sh2 * C0 + 0.5);
    } else if (hasRGBColor) {
      const rInfo = propOffsets["red"];
      const gInfo = propOffsets["green"];
      const bInfo = propOffsets["blue"];
      // Handle both uchar (0-255) and float (0-1)
      if (rInfo.type === "uchar" || rInfo.type === "uint8") {
        colors[i * 3] = dataView.getUint8(base + rInfo.offset) / 255;
        colors[i * 3 + 1] = dataView.getUint8(base + gInfo.offset) / 255;
        colors[i * 3 + 2] = dataView.getUint8(base + bInfo.offset) / 255;
      } else {
        colors[i * 3] = dataView.getFloat32(base + rInfo.offset, true);
        colors[i * 3 + 1] = dataView.getFloat32(base + gInfo.offset, true);
        colors[i * 3 + 2] = dataView.getFloat32(base + bInfo.offset, true);
      }
    } else {
      // Default gray
      colors[i * 3] = 0.7;
      colors[i * 3 + 1] = 0.7;
      colors[i * 3 + 2] = 0.7;
    }

    // Opacity
    if (hasOpacity && opacities) {
      const rawOpacity = dataView.getFloat32(
        base + propOffsets["opacity"].offset,
        true,
      );
      opacities[i] = sigmoid(rawOpacity);
    }
  }

  return {
    positions,
    colors,
    opacities,
    count: vertexCount,
    fileSize: buffer.byteLength,
  };
}

/**
 * Fetch and parse a PLY file from a URL (e.g., Tauri asset protocol URL).
 * Uses streaming fetch for large files.
 */
export async function loadPlyFromUrl(url: string): Promise<PlyData> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch PLY: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  return parsePlyBuffer(buffer);
}

/**
 * Parse a PLY file from a Uint8Array (e.g., from Tauri fs readFile).
 */
export function loadPlyFromBytes(bytes: Uint8Array): PlyData {
  return parsePlyBuffer(bytes.buffer);
}
