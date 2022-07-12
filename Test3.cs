MCC.LoadMacros(new Test());

//Script Extensions
 
public class Test : Macros
{
  public override bool OnKeyDown(Key key, bool repeat)
  {
    if (key == Key.R)
    {
      KeyDown(Key.G, Key.B, Key.L, Key.J, Key.H);
      KeyUp(Key.G, Key.B, Key.L, Key.J, Key.H);
                    
      KeyDown(Key.Enter);
      KeyUp(Key.Enter);
      return true;
    }
    return false;
  }
}
