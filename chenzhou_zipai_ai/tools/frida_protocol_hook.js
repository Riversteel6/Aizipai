// Read-only Frida probe: log recv/send buffers that contain known APK protocol field names.
const patterns = [
  "playerholdcards", "curr_card", "out_cards", "canpeng", "canchi", "canhu",
  "showGuo", "huxi", "hupai", "precardval", "isoutcard", "huzi_cardval",
  "card_num", "gamestate", "zhuang", "zuozhuang", "chairId", "douzhuangChairId"
];

function tryRead(ptrValue, length) {
  const size = Math.min(length, 4096);
  if (size <= 0) return "";
  try {
    if (typeof ptrValue.readUtf8String === "function") {
      return ptrValue.readUtf8String(size) || "";
    }
    return Memory.readUtf8String(ptrValue, size) || "";
  } catch (_) {
    try {
      return hexdump(ptrValue, { length: Math.min(size, 256), ansi: false });
    } catch (_) {
      return "";
    }
  }
}

function findExport(name) {
  if (typeof Module.findGlobalExportByName === "function") {
    return Module.findGlobalExportByName(name);
  }
  if (typeof Module.getGlobalExportByName === "function") {
    try {
      return Module.getGlobalExportByName(name);
    } catch (_) {
      return null;
    }
  }
  if (typeof Module.findExportByName === "function") {
    return Module.findExportByName(null, name);
  }
  if (typeof Module.getExportByName === "function") {
    try {
      return Module.getExportByName(null, name);
    } catch (_) {
      return null;
    }
  }
  return null;
}

function interesting(text) {
  return patterns.some(p => text.indexOf(p) !== -1);
}

function hookIo(name, bufIndex, lenIndex, after) {
  const addr = findExport(name);
  if (!addr) return;
  Interceptor.attach(addr, {
    onEnter(args) {
      this.buf = args[bufIndex];
      this.len = args[lenIndex].toInt32();
    },
    onLeave(retval) {
      const actual = after ? retval.toInt32() : this.len;
      if (actual <= 0) return;
      const text = tryRead(this.buf, actual);
      if (interesting(text)) {
        send({ type: name, size: actual, text: text });
      }
    }
  });
}

hookIo("recv", 1, 2, true);
hookIo("send", 1, 2, false);
hookIo("read", 1, 2, true);
hookIo("write", 1, 2, false);
send({ type: "ready", patterns: patterns });
