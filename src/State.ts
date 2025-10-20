import { atom } from "recoil";

export const wandbURLAtom = atom({
  key: "wandbURLAtom",
  default: "https://wandb.ai",
});

export const reportModeAtom = atom({
  key: "reportModeAtom",
  default: false,  // false = launcher mode, true = embedded report mode
});
