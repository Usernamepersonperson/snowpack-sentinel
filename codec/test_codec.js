// Runs the shared vectors through the JS decoder: node codec/test_codec.js
var fs = require("fs");
var path = require("path");
var assert = require("assert");
var decodeUplink = require("./ttn_chirpstack.js").decodeUplink;

var vectors = JSON.parse(
  fs.readFileSync(path.join(__dirname, "..", "examples", "vectors.json"), "utf8")
);
var failed = 0;
vectors.forEach(function (v) {
  var bytes = [];
  for (var i = 0; i < v.hex.length; i += 2) { bytes.push(parseInt(v.hex.substr(i, 2), 16)); }
  var res = decodeUplink({ bytes: bytes, fPort: 1 });
  try {
    if (v.error) {
      assert.ok(res.errors && res.errors.length > 0, "expected an error");
    } else {
      assert.deepStrictEqual(res.data, v.data);
    }
    console.log("ok   " + v.name);
  } catch (e) {
    failed += 1;
    console.log("FAIL " + v.name + ": " + e.message);
  }
});
console.log(vectors.length - failed + "/" + vectors.length + " vectors passed");
process.exit(failed ? 1 : 0);
