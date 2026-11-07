/*
 * snowpack-sentinel uplink decoder (16-byte frame, see snowpack/codec.py).
 * Signature works for The Things Stack and ChirpStack v4: decodeUplink(input).
 * Plain ES5, no imports.
 */
function decodeUplink(input) {
  var b = input.bytes;
  var warnings = [];
  if (!b || b.length !== 16) {
    return { data: {}, warnings: warnings, errors: ["expected 16 bytes, got " + (b ? b.length : 0)] };
  }
  function i16(o) {
    var v = (b[o] << 8) | b[o + 1];
    return v & 0x8000 ? v - 0x10000 : v;
  }
  function temp(o) {
    var v = i16(o);
    return v === -32768 ? null : v / 100;
  }
  function u8(o, f) {
    return b[o] === 0xff ? null : f(b[o]);
  }
  var w0 = (b[0] << 8) | b[1];
  var hs = w0 & 0x1fff;
  var fl = w0 >> 13;
  var flags = {
    low_battery: (fl & 1) === 1,
    sensor_fault: (fl & 2) === 2,
    event: (fl & 4) === 4
  };
  if (flags.low_battery) { warnings.push("low battery"); }
  if (flags.sensor_fault) { warnings.push("sensor fault flag set"); }
  var data = {
    hs_mm: hs === 0x1fff ? null : hs,
    ta_c: temp(2),
    tss_c: temp(4),
    t25_c: temp(6),
    t50_c: temp(8),
    t100_c: temp(10),
    wind_mean_ms: u8(12, function (v) { return v * 2 / 10; }),
    wind_gust_ms: u8(13, function (v) { return v * 2 / 10; }),
    wind_dir_deg: b[14] >= 180 ? null : b[14] * 2,
    battery_mv: u8(15, function (v) { return 2000 + v * 10; }),
    flags: flags
  };
  return { data: data, warnings: warnings };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { decodeUplink: decodeUplink };
}
